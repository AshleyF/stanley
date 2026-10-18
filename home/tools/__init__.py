"""Device tools. Each module owns its schema and its call."""

from .garage import Garage
from .lights import Lights

def load_tools():
    return [Lights(), Garage()]

def find_tool(tools, name):
    return next((tool for tool in tools if tool.handles(name)), None)
