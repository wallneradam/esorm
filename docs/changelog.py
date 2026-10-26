import os
import re

import requests

# A heading line of a release body, e.g. "## What's Changed"
_HEADING = re.compile(r'^(#{1,6})(?= )')
_FENCE = re.compile(r'^\s*(```|~~~)')


def _demote_headings(body: str) -> str:
    """
    Move the headings of a release body below the heading of the release, so the top level
    headings of the body become level 3 headings

    :param body: The Markdown body of the release
    :return: The body with demoted headings
    """
    lines = body.split('\n')
    # The headings outside of code blocks
    headings = {}
    in_code = False
    for i, line in enumerate(lines):
        if _FENCE.match(line):
            in_code = not in_code
        elif not in_code and (m := _HEADING.match(line)):
            headings[i] = len(m[1])
    if not headings:
        return body

    shift = 3 - min(headings.values())
    for i, level in headings.items():
        lines[i] = '#' * min(level + shift, 6) + lines[i][level:]
    return '\n'.join(lines)


def _fetch_releases(repository: str) -> list:
    """
    Fetch all releases of a public GitHub repository

    :param repository: The GitHub repository in the format 'username/repo'
    :return: The list of releases, the newest first
    """
    headers = {'Accept': 'application/vnd.github+json'}
    # Unauthenticated requests are rate limited, a token helps on shared build servers
    token = os.environ.get('GITHUB_TOKEN')
    if token:
        headers['Authorization'] = f'Bearer {token}'

    releases = []
    url = f"https://api.github.com/repos/{repository}/releases?per_page=100"
    while url:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        releases.extend(response.json())
        url = response.links.get('next', {}).get('url')
    return releases


def generate_changelog(repository, file_path="changelog.md"):
    """
    Generate a changelog in Markdown format from GitHub releases of a public repository.
    The changelog will have a single main title 'Changelog' and each release as a subsection.

    If the releases can't be fetched, the changelog only links to the releases page, so the
    documentation can still be built.

    :param repository: The GitHub repository in the format 'username/repo'.
    :param file_path: The path of the changelog file.
    """
    changelog = ["# Changelog\n"]

    try:
        releases = _fetch_releases(repository)
    except requests.RequestException as e:
        print(f"WARNING: Could not fetch the releases of {repository}: {e}")
        releases = None

    if releases is None:
        changelog.append(f"See the [releases on GitHub](https://github.com/{repository}/releases).\n")
    else:
        for release in releases:
            title = release["name"] or release["tag_name"]
            date = release["published_at"].split("T")[0]
            body = _demote_headings((release["body"] or "").replace('\r\n', '\n').strip())
            changelog.append(f"## {title}\n\n*Released on {date}*\n\n{body}\n")

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write("\n".join(changelog))
