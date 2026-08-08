import smtplib
from email.message import EmailMessage
from pathlib import Path

import httpx
import pytest
from app import main as gateway
from app.config import Settings
from app.email import MAX_ATTACHMENT_BYTES, AttachmentTooLargeError, build_message
from app.main import create_app
from app.sandbox import FakeSandbox, SessionManager


def test_build_message_attaches_the_artifact():
    message = build_message(
        "copilot@example.com",
        "analyst@example.com",
        "Data Copilot: deck.pptx",
        "Here is the deck.",
        "deck.pptx",
        b"PK\x03\x04 fake",
    )
    assert message["From"] == "copilot@example.com"
    assert message["To"] == "analyst@example.com"
    assert message["Subject"] == "Data Copilot: deck.pptx"
    payload = message.get_payload()
    assert isinstance(payload, list)
    body_part, attachment = payload
    assert isinstance(body_part, EmailMessage)
    assert isinstance(attachment, EmailMessage)
    assert body_part.get_content() == "Here is the deck.\n"
    assert attachment.get_filename() == "deck.pptx"
    assert attachment.get_content_type() == (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    )
    assert attachment.get_payload(decode=True) == b"PK\x03\x04 fake"


def test_build_message_guards_attachment_size():
    with pytest.raises(AttachmentTooLargeError):
        build_message("a@b.c", "d@e.f", "s", "", "big.pdf", b"x" * (MAX_ATTACHMENT_BYTES + 1))


class FakeSMTP:
    """Captures deliveries; failures raise like smtplib does."""

    deliveries: list[EmailMessage] = []
    fail_with: Exception | None = None

    def __init__(self, host: str, port: int, timeout: int = 15) -> None:
        self.host = host
        self.port = port

    def __enter__(self) -> "FakeSMTP":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def starttls(self) -> None:
        return None

    def login(self, user: str, password: str) -> None:
        return None

    def send_message(self, message: EmailMessage) -> None:
        if FakeSMTP.fail_with is not None:
            raise FakeSMTP.fail_with
        FakeSMTP.deliveries.append(message)


@pytest.fixture
def email_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    FakeSMTP.deliveries = []
    FakeSMTP.fail_with = None
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    settings = Settings(
        sessions_dir=str(tmp_path / "sessions"),
        smtp_host="smtp.example.com",
        email_from="copilot@example.com",
        email_recipient="analyst@example.com",
    )
    manager = SessionManager(
        settings,
        sandbox_factory=lambda sid: FakeSandbox(sid, tmp_path / "workspaces" / sid),
    )
    return create_app(settings, manager), manager


async def _session_with_artifact(client: httpx.AsyncClient, manager: SessionManager) -> str:
    created = await client.post("/api/sessions")
    session_id = created.json()["id"]
    session = manager.get(session_id)
    await session.sandbox.upload("deck.pptx", b"PK\x03\x04 fake deck")
    return session_id


@pytest.mark.asyncio
async def test_email_endpoint_sends_the_artifact(email_app):
    application, manager = email_app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://test"
    ) as client:
        session_id = await _session_with_artifact(client, manager)
        response = await client.post(
            f"/api/sessions/{session_id}/artifacts/deck.pptx/email",
            json={"subject": "Q3 deck", "message": "Worth a look."},
        )

    assert response.status_code == 200
    assert response.json() == {
        "sent": True,
        "recipient": "analyst@example.com",
        "name": "deck.pptx",
    }
    assert len(FakeSMTP.deliveries) == 1
    delivered = FakeSMTP.deliveries[0]
    assert delivered["To"] == "analyst@example.com"
    assert delivered["Subject"] == "Q3 deck"
    delivered_parts = delivered.get_payload()
    assert isinstance(delivered_parts, list)
    delivered_attachment = delivered_parts[1]
    assert isinstance(delivered_attachment, EmailMessage)
    assert delivered_attachment.get_filename() == "deck.pptx"


@pytest.mark.asyncio
async def test_email_endpoint_defaults_the_subject(email_app):
    application, manager = email_app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://test"
    ) as client:
        session_id = await _session_with_artifact(client, manager)
        response = await client.post(
            f"/api/sessions/{session_id}/artifacts/deck.pptx/email", json={}
        )

    assert response.status_code == 200
    assert FakeSMTP.deliveries[0]["Subject"] == "Data Copilot: deck.pptx"


@pytest.mark.asyncio
async def test_email_endpoint_503_when_unconfigured(tmp_path: Path):
    settings = Settings(sessions_dir=str(tmp_path / "sessions"))
    manager = SessionManager(
        settings,
        sandbox_factory=lambda sid: FakeSandbox(sid, tmp_path / "workspaces" / sid),
    )
    application = create_app(settings, manager)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://test"
    ) as client:
        created = await client.post("/api/sessions")
        session_id = created.json()["id"]
        response = await client.post(
            f"/api/sessions/{session_id}/artifacts/deck.pptx/email", json={}
        )
        config = await client.get("/api/config/email")

    assert response.status_code == 503
    assert config.json() == {"configured": False, "recipient": ""}


@pytest.mark.asyncio
async def test_email_endpoint_404_for_missing_artifact(email_app):
    application, manager = email_app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://test"
    ) as client:
        created = await client.post("/api/sessions")
        session_id = created.json()["id"]
        response = await client.post(
            f"/api/sessions/{session_id}/artifacts/nope.pdf/email", json={}
        )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_email_endpoint_502_on_smtp_failure(email_app):
    FakeSMTP.fail_with = smtplib.SMTPAuthenticationError(535, b"bad credentials")
    application, manager = email_app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://test"
    ) as client:
        session_id = await _session_with_artifact(client, manager)
        response = await client.post(
            f"/api/sessions/{session_id}/artifacts/deck.pptx/email", json={}
        )

    assert response.status_code == 502
    assert "SMTP delivery failed" in response.json()["detail"]


@pytest.mark.asyncio
async def test_email_endpoint_413_for_oversized_artifact(email_app):
    application, manager = email_app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://test"
    ) as client:
        created = await client.post("/api/sessions")
        session_id = created.json()["id"]
        session = manager.get(session_id)
        await session.sandbox.upload("huge.pdf", b"x" * (MAX_ATTACHMENT_BYTES + 1))
        response = await client.post(
            f"/api/sessions/{session_id}/artifacts/huge.pdf/email", json={}
        )

    assert response.status_code == 413
    assert not FakeSMTP.deliveries


@pytest.mark.asyncio
async def test_email_config_endpoint_reports_the_recipient(email_app):
    application, _manager = email_app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://test"
    ) as client:
        response = await client.get("/api/config/email")

    assert response.json() == {"configured": True, "recipient": "analyst@example.com"}


def test_email_endpoint_is_registered(email_app):
    application, _manager = email_app
    paths = {route.path for route in application.routes}
    assert "/api/sessions/{session_id}/artifacts/{name:path}/email" in paths
    assert gateway.EmailRequest is not None
