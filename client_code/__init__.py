# Client code entry point
import anvil.server

# Connect to the server
anvil.server.connect("PSL-APP")

# Import the main form
from .MainForm import MainForm

# Set the main form as the startup form
anvil.server.wait_forever()
