"""Garage doors. Kept, and not loaded.

On 18 October 2026 a login with the account credentials reached
https://partner-identity.myq-cloud.com/connect/token and got 401.
Chamberlain refuses this unofficial client. The official app and Tesla still
work, because Tesla is a partner. A local opener replaces this file only.
load_tools() does not return Garage, so a session never offers set_door.
"""

import os

from .lights import find_named, listing

class Garage:
    name = "set_door"

    def handles(self, name):
        return name == self.name

    def instructions(self, home):
        doors = home["doors"]
        if not doors:
            return ""
        return (
            "You can open or close these doors:\n"
            f"{listing(doors)}\n"
            "Call set_door only when they clearly ask to open or close one. Use its name."
        )

    def schemas(self, home):
        doors = home["doors"]
        if not doors:
            return []
        return [{
            "type": "function",
            "name": self.name,
            "description": "Open or close one listed door.\n" + listing(doors),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": [door["name"] for door in doors]},
                    "action": {"type": "string", "enum": ["open", "close"]},
                },
                "required": ["name", "action"],
            },
        }]

    async def call(self, _name, args, home):
        door = find_named(home["doors"], args.get("name"))
        action = str(args.get("action") or "").casefold()
        if door is None:
            print(f"\nno door named {args.get('name')}", flush=True)
            return {"error": f"no door named {args.get('name')}"}
        if action not in ("open", "close"):
            return {"error": "action must be open or close"}
        username, password = os.environ.get("MYQ_USERNAME"), os.environ.get("MYQ_PASSWORD")
        if not username or not password:
            print("\nMyQ login is not set", flush=True)
            return {"error": "Set MYQ_USERNAME and MYQ_PASSWORD in .env"}
        held = None
        try:
            held = await self.cover(username, password, door["name"], len(home["doors"]))
            if action == "open":
                await held.open(wait_for_state=False)
            else:
                await held.close(wait_for_state=False)
            print(f"\n{door['name']} {action}", flush=True)
            return {"name": door["name"], "action": action}
        except Exception as exc:
            text = str(exc).strip() or type(exc).__name__
            print(f"\n{text}", flush=True)
            return {"error": text}
        finally:
            if held is not None:
                await held.shutdown()

    async def cover(self, username, password, name, configured):
        import pymyq
        from aiohttp import ClientSession
        websession = ClientSession()
        try:
            account = await pymyq.login(username, password, websession)
            covers = list(account.covers.values())
            wanted = name.casefold()
            chosen = next((cover for cover in covers if (getattr(cover, "name", "") or "").casefold() == wanted), None)
            if chosen is None and len(covers) == 1 and configured == 1:
                chosen = covers[0]
            if chosen is None:
                names = ", ".join(getattr(cover, "name", "") or "unnamed" for cover in covers) or "none"
                raise RuntimeError(f"no MyQ door named {name} (account has {names})")
            return _Held(chosen, websession)
        except Exception:
            await websession.close()
            raise

class _Held:
    """Keep the HTTP session alive until the door command finishes."""

    def __init__(self, cover, websession):
        self.cover, self.websession = cover, websession

    async def open(self, **kwargs):
        await self.cover.open(**kwargs)

    async def close(self, **kwargs):
        await self.cover.close(**kwargs)

    async def shutdown(self):
        await self.websession.close()
