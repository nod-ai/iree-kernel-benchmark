from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.view import view_config, view_defaults
from pyramid.response import Response
from github import Github

ENDPOINT = "webhook"

@view_defaults(
    route_name=ENDPOINT, renderer="json", request_method="POST"
)
class PayloadView:
    """
    Handles incoming GitHub webhook payloads.
    The view is triggered only for JSON payloads sent via POST requests.
    """
    def __init__(self, request):
        self.request = request
        self.payload = self.request.json

    @view_config(header="X-Github-Event:push")
    def payload_push(self):
        """Handles push events."""
        print("Number of commits in push:", len(self.payload['commits']))
        return Response("success")

    @view_config(header="X-Github-Event:pull_request")
    def payload_pull_request(self):
        """Handles pull request events."""
        print("Pull Request action:", self.payload['action'])
        print("Number of commits in PR:", self.payload['pull_request']['commits'])
        return Response("success")

    @view_config(header="X-Github-Event:ping")
    def payload_else(self):
        """Handles GitHub's ping event when a webhook is created."""
        print("Webhook created with ID {}!".format(self.payload["hook"]["id"]))
        return {"status": 200}

if __name__ == "__main__":
    config = Configurator()
    config.add_route(ENDPOINT, "/{}".format(ENDPOINT))
    config.scan()
    app = config.make_wsgi_app()
    server = make_server("127.0.0.1", 2500, app)
    server.serve_forever()
