import asyncio
import websockets
import sys

print("DEBUG: ws_minimal starting")

async def echo(websocket, path):
    async for message in websocket:
        await websocket.send(message)

start_server = websockets.serve(echo, "localhost", 32051)

print("DEBUG: Server starting...")
asyncio.get_event_loop().run_until_complete(start_server)
print("DEBUG: Server running")
asyncio.get_event_loop().run_forever()
