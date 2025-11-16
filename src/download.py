"""
Lichess Database Download Utility
----------------------------------
Downloads PGN files from Lichess database with resume support and progress tracking.
"""

import os
import sys
import argparse
import requests
from tqdm import tqdm
from pathlib import Path
from urllib.parse import urlparse


def construct_lichess_url(date_str: str = None) -> str:
    """
    Construct Lichess database URL from date string.
    
    Args:
        date_str: Date string in format "YYYY-MM" (e.g., "2025-10")
                 If None, returns base URL pattern
    
    Returns:
        Full URL to Lichess database file
    """
    if date_str:
        return f"https://database.lichess.org/standard/lichess_db_standard_rated_{date_str}.pgn.zst"
    return "https://database.lichess.org/standard/"


def validate_url(url: str) -> bool:
    """Validate that URL is a Lichess database URL."""
    return "database.lichess.org" in url and ("pgn" in url or url.endswith("/standard/"))


def download_file(url: str, output_path: str, resume: bool = True) -> bool:
    """
    Download a file from URL with resume support.
    
    Args:
        url: URL to download from
        output_path: Path to save file
        resume: Whether to resume interrupted downloads
    
    Returns:
        True if download successful, False otherwise
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Check if file exists and get size for resume
    existing_size = 0
    if os.path.exists(output_path) and resume:
        existing_size = os.path.getsize(output_path)
        if existing_size > 0:
            print(f"Resuming download from {existing_size:,} bytes")
    
    # Set up headers for resume
    headers = {}
    if existing_size > 0 and resume:
        headers['Range'] = f'bytes={existing_size}-'
    
    try:
        # Start download
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        # Check if server supports resume
        if existing_size > 0 and response.status_code == 206:
            # Partial content - resume download
            mode = 'ab'
            total_size = int(response.headers.get('Content-Range', '').split('/')[-1])
        elif existing_size > 0:
            # Server doesn't support resume, restart download
            print("Server doesn't support resume, restarting download...")
            existing_size = 0
            mode = 'wb'
            total_size = int(response.headers.get('Content-Length', 0))
        else:
            # New download
            mode = 'wb'
            total_size = int(response.headers.get('Content-Length', 0))
        
        # Download with progress bar
        with open(output_path, mode) as f:
            with tqdm(
                total=total_size,
                initial=existing_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=os.path.basename(output_path)
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"\nDownload complete: {output_path}")
        print(f"File size: {os.path.getsize(output_path):,} bytes")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\nDownload interrupted. File saved to: {output_path}")
        print(f"Run the same command again to resume download.")
        return False


def parse_date_or_url(input_str: str) -> str:
    """
    Parse input as either date string or full URL.
    
    Args:
        input_str: Either date string (YYYY-MM) or full URL
    
    Returns:
        Full URL
    """
    # Check if it's a URL
    if input_str.startswith('http://') or input_str.startswith('https://'):
        return input_str
    
    # Check if it's a date string (YYYY-MM)
    if len(input_str) == 7 and input_str[4] == '-':
        try:
            year, month = input_str.split('-')
            int(year)
            int(month)
            if 1 <= int(month) <= 12:
                return construct_lichess_url(input_str)
        except ValueError:
            pass
    
    # If neither, assume it's a date string and try anyway
    return construct_lichess_url(input_str)


def main():
    parser = argparse.ArgumentParser(
        description="Download PGN files from Lichess database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download by date
  python src/download.py 2025-10
  
  # Download by full URL
  python src/download.py https://database.lichess.org/standard/lichess_db_standard_rated_2025-10.pgn.zst
  
  # Specify output directory
  python src/download.py 2025-10 --output data/
        """
    )
    parser.add_argument("url_or_date", type=str,
                       help="Lichess URL or date string (YYYY-MM, e.g., '2025-10')")
    parser.add_argument("--output", "-o", type=str, default="data/",
                       help="Output directory (default: data/)")
    parser.add_argument("--no-resume", action="store_true",
                       help="Don't resume interrupted downloads")
    
    args = parser.parse_args()
    
    # Parse URL or date
    url = parse_date_or_url(args.url_or_date)
    
    if not validate_url(url):
        print(f"Warning: URL doesn't appear to be a Lichess database URL: {url}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Extract filename from URL
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    if not filename:
        # If no filename in URL, try to construct from date
        if args.url_or_date and '-' in args.url_or_date:
            filename = f"lichess_db_standard_rated_{args.url_or_date}.pgn.zst"
        else:
            filename = "lichess_database.pgn.zst"
    
    # Construct output path
    output_path = os.path.join(args.output, filename)
    
    print(f"Downloading: {url}")
    print(f"Output: {output_path}")
    print()
    
    # Download file
    success = download_file(url, output_path, resume=not args.no_resume)
    
    if success:
        print(f"\n✅ Successfully downloaded to: {output_path}")
        sys.exit(0)
    else:
        print(f"\n❌ Download failed or interrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()

