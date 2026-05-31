import os
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_photos(matches: dict[str, list[str]], smtp_config: dict) -> None:
    """Send each matched employee their event photos as email attachments."""
    host      = smtp_config["host"]
    port      = smtp_config["port"]
    user      = smtp_config["user"]
    password  = smtp_config["password"]
    from_addr = smtp_config.get("from_addr", user)

    with _connect(host, port, user, password) as server:
        for email, photo_paths in matches.items():
            msg = _build_message(from_addr, email, photo_paths)
            server.sendmail(from_addr, email, msg.as_string())
            print(f"  ✓ sent {len(photo_paths)} photo(s) to {email}")


def _connect(host: str, port: int, user: str, password: str):
    """Return an authenticated SMTP connection. Port 465 = SSL, else STARTTLS."""
    if port == 465:
        server = smtplib.SMTP_SSL(host, port)
    else:
        server = smtplib.SMTP(host, port)
        server.starttls()
    server.login(user, password)
    return server


def _build_message(from_addr: str, to_addr: str, photo_paths: list[str]) -> MIMEMultipart:
    msg = MIMEMultipart()
    msg["From"]    = from_addr
    msg["To"]      = to_addr
    msg["Subject"] = "Your event photos"

    msg.attach(MIMEText(
        f"Hi,\n\nWe found {len(photo_paths)} photo(s) of you from the event. "
        "See the attachments!\n\nBest regards",
        "plain",
    ))

    for path in photo_paths:
        with open(path, "rb") as f:
            img = MIMEImage(f.read())
        img.add_header("Content-Disposition", "attachment", filename=os.path.basename(path))
        msg.attach(img)

    return msg
