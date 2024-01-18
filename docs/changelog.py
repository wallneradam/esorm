import requests


def generate_changelog(repository, file_path="changelog.rst"):
    """
    Generate a changelog in reStructuredText format from GitHub releases of a public repository.
    The changelog will have a single main title 'Changelog' and each release as a subsection.

    :param repository: The GitHub repository in the format 'username/repo'.
    :param file_path: The path of the changelog file.
    """
    # GitHub API URL for the releases of the repository
    api_url = f"https://api.github.com/repos/{repository}/releases"

    # Send a request to the GitHub API
    response = requests.get(api_url)

    # Check if the response is successful
    if response.status_code == 200:
        releases = response.json()

        # Main title for the changelog
        changelog = ["Changelog\n=========\n\n"]

        # Process each release
        for release in releases:
            title = release["name"]
            date = release["published_at"].split("T")[0]
            body = release["body"].replace('\r\n', '\n').strip()

            # Format each release as a subsection
            release_entry = f"{title}\n{'-' * len(title)}\n*Released on*: {date}\n\n{body}\n\n"
            changelog.append(release_entry)

        # Join all entries into a single string
        changelog = "\n".join(changelog)

        # Save changelog to file
        with open(file_path, 'w') as file:
            file.write(changelog)
    else:
        response.raise_for_status()
