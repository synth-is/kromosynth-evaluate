import asyncio
import websockets
import json
import argparse

async def socket_server(websocket, path):
    # Wait for the first message and determine its type
    message = await websocket.recv()

    if isinstance(message, bytes):
        # Received binary message (assume it's an audio buffer)
        audio_data = message
        print('Standalone audio data received')
        # Process the audio data...
        response = {'status': 'received standalone audio'}
        await websocket.send(json.dumps(response))
    else:
        # Received text message (assume it's a JSON message)
        try:
            json_data = json.loads(message)
            print('JSON data received:', json_data)

            # Now wait for the binary message (audio buffer)
            audio_data = await websocket.recv()
            if not isinstance(audio_data, bytes):
                print("Expected a binary message, but didn't receive one.")
                response = {'status': 'error', 'message': 'Expected binary audio after JSON'}
                await websocket.send(json.dumps(response))
            else:
                print('Audio data received after JSON')
                # Process the audio data...
                response = {'status': 'received JSON and audio'}
                await websocket.send(json.dumps(response))

        except json.JSONDecodeError:
            # Message is not valid JSON
            response = {'status': 'error', 'message': 'Invalid JSON'}
            await websocket.send(json.dumps(response))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run a WebSocket server.')
parser.add_argument('--host', type=str, default='localhost', help='Host to run the WebSocket server on.')
parser.add_argument('--port', type=int, default=8080, help='Port number to run the WebSocket server on.')
args = parser.parse_args()

print('Starting WebSocket server at ws://{}:{}'.format(args.host, args.port))

# Start the WebSocket server with supplied command line arguments
start_server = websockets.serve(socket_server, args.host, args.port)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()