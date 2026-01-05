def get_url(
    host: str, port: int, endpoint: str | None = None, protocol: str = "http://"
):
    """
    Get the URL for a given host and port; if port is negative, skip it.
    Cleans the host and endpoint strings.
    If the host starts with https://, that protocol will be preserved.
    """
    # Detect and preserve protocol from host if present
    detected_protocol = protocol
    if host.startswith("https://"):
        detected_protocol = "https://"
    elif host.startswith("http://"):
        detected_protocol = "http://"

    # Remove http:// or https:// from host if present
    host_str = host.replace("http://", "").replace("https://", "")
    port_str = f":{port}" if int(port) >= 0 else ""
    endpoint_str = f"/{endpoint.lstrip('/')}" if endpoint else ""
    return f"{detected_protocol}{host_str}{port_str}{endpoint_str}"
