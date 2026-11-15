"""Health and root endpoint tests."""


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert "Safety Driver" in body["service"]
    assert body["docs_url"] == "/docs"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["status"] in {"healthy", "degraded"}
    assert "models" in body
    assert "mediapipe_face_landmarker" in body["models"]
