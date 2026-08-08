"""Artifact email delivery over SMTP.

Sending lives on the gateway, not in the sandbox: sandbox egress is disabled
by design, and the SMTP password never leaves the backend process. The
recipient is fixed in settings so the endpoint cannot be used as an open
relay.
"""

from __future__ import annotations

import mimetypes
import smtplib
from email.message import EmailMessage

from .config import Settings

# Most SMTP servers reject messages past ~25 MB; keep headroom for base64.
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024


class AttachmentTooLargeError(ValueError):
    """The artifact exceeds the outbound size guard."""


def build_message(
    sender: str,
    recipient: str,
    subject: str,
    body: str,
    filename: str,
    content: bytes,
) -> EmailMessage:
    if len(content) > MAX_ATTACHMENT_BYTES:
        raise AttachmentTooLargeError(
            f"{filename} is {len(content) / (1024 * 1024):.1f} MB; "
            f"the email limit is {MAX_ATTACHMENT_BYTES // (1024 * 1024)} MB"
        )
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body or "Sent from Data Copilot.")
    media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    maintype, subtype = media_type.split("/", 1)
    message.add_attachment(content, maintype=maintype, subtype=subtype, filename=filename)
    return message


def send_message(settings: Settings, message: EmailMessage) -> None:
    """Deliver via SMTP. Blocking; call from a worker thread."""
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15) as smtp:
        if settings.smtp_starttls:
            smtp.starttls()
        if settings.smtp_user:
            smtp.login(settings.smtp_user, settings.smtp_password)
        smtp.send_message(message)
