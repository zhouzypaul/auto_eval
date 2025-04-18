def get_url(
    host: str, port: int, endpoint: str | None = None, protocol: str = "http://"
):
    """
    Get the URL for a given host and port; if port is negative, skip it.
    Cleans the host and endpoint strings
    """
    # Remove http:// or https:// from host if present
    host_str = host.replace("http://", "").replace("https://", "")
    port_str = f":{port}" if int(port) >= 0 else ""
    endpoint_str = f"/{endpoint.lstrip('/')}" if endpoint else ""
    return f"{protocol}{host_str}{port_str}{endpoint_str}"
