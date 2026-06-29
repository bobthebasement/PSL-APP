# Server entry point
import anvil.server

# Import the app module to initialize everything
from . import __init__

# Start the server - DO NOT call anvil.server.connect() here
# Anvil will handle the connection automatically
