import argparse
import sys

from photo_mailer.face_db import build_db, load_db, save_db
from photo_mailer.mailer import send_photos
from photo_mailer.matcher import match_photos
from photo_mailer.scraper import mock_employees, scrape_employees


def main():
    parser = argparse.ArgumentParser(description="Match faces in event photos and email results.")
    parser.add_argument("--url",        help="Company team page URL (omit with --mock)")
    parser.add_argument("--mock",       action="store_true", help="Use hardcoded test employees")
    parser.add_argument("--photos-dir", required=True,       help="Folder with event photos")
    parser.add_argument("--db-path",    default="face_db.pkl", help="Path to save/load face DB")
    parser.add_argument("--threshold",  type=float, default=0.60, help="Match similarity threshold")
    parser.add_argument("--dry-run",    action="store_true", help="Print matches, skip sending emails")
    parser.add_argument("--smtp-host",  default="smtp.gmail.com")
    parser.add_argument("--smtp-port",  type=int, default=465)
    parser.add_argument("--smtp-user",  default="")
    parser.add_argument("--smtp-pass",  default="")
    parser.add_argument("--smtp-from",  default="", help="Sender address (defaults to --smtp-user)")
    args = parser.parse_args()

    if not args.mock and not args.url:
        parser.error("Provide --url or --mock")

    # Step 1: get employees
    print("Step 1: loading employees...")
    employees = mock_employees() if args.mock else scrape_employees(args.url)
    print(f"  {len(employees)} employee(s) loaded")

    # Step 2: build or reuse face DB
    print("Step 2: building face DB...")
    db = build_db(employees)
    save_db(db, args.db_path)

    # Step 3: match event photos
    print("Step 3: matching event photos...")
    matches = match_photos(args.photos_dir, db, threshold=args.threshold)
    print(f"  Matches found for {len(matches)} employee(s):")
    for email, paths in matches.items():
        print(f"    {email}: {len(paths)} photo(s)")

    if not matches:
        print("No matches — nothing to send.")
        sys.exit(0)

    # Step 4: send emails
    if args.dry_run:
        print("Dry run — skipping email send.")
        return

    print("Step 4: sending emails...")
    smtp_config = {
        "host":      args.smtp_host,
        "port":      args.smtp_port,
        "user":      args.smtp_user,
        "password":  args.smtp_pass,
        "from_addr": args.smtp_from or args.smtp_user,
    }
    send_photos(matches, smtp_config)
    print("Done.")


if __name__ == "__main__":
    main()
