"""Device tools. Each module owns its schema and its call."""

from .lights import Lights
from .shower import Shower
from .timers import Timers

def load_tools():
    # Garage stays in garage.py and is not offered. MyQ answered 401 on
    # https://partner-identity.myq-cloud.com/connect/token for this client.
    return [Lights(), Shower(), Timers()]

def find_tool(tools, name):
    return next((tool for tool in tools if tool.handles(name)), None)
