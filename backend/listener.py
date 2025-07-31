from webhook import WorkflowListener, WaveUpdateListener
from storage import DatabaseClient, DirectoryClient
from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.view import view_config, view_defaults
from pyramid.response import Response
import json
from dotenv import load_dotenv
import os

load_dotenv()
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = "benchmarkcache"

db_client = DatabaseClient(connection_string)
dir_client = DirectoryClient(connection_string, container_name)

workflow_client = WorkflowListener(db_client, dir_client)
wave_update_client = WaveUpdateListener(db_client, dir_client)

ENDPOINT = "webhook"


@view_defaults(route_name=ENDPOINT, renderer="json", request_method="POST")
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
        print("Number of commits in push:", len(self.payload["commits"]))
        return Response("success")

    @view_config(header="X-Github-Event:pull_request")
    def payload_pull_request(self):
        """Handles pull request events."""
        print("Pull Request action:", self.payload["action"])
        print("Number of commits in PR:", self.payload["pull_request"]["commits"])
        wave_update_client.handle_pr_payload(self.payload)
        return Response("success")

    @view_config(header="X-Github-Event:workflow_run")
    def payload_workflow_run(self):
        workflow_client.handle_workflow_run_payload(self.payload)
        return Response("success")

    @view_config(header="X-Github-Event:workflow_job")
    def payload_workflow_job(self):
        workflow_client.handle_workflow_job_payload(self.payload)
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
