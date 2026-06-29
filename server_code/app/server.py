# Server entry point
import anvil.server

# Import the app module to initialize everything
from . import app

# Start the server
anvil.server.wait_forever()
