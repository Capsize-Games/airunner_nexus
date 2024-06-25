import asyncio
import websockets

# Define a callback function to handle incoming WebSocket messages
async def handle_websocket(websocket, path):
    try:
        while True:
            message = await websocket.recv()
            print(f"Received message: >{message}<")

            # XXX: Do some stuff here with the message.

            response = "Pong!"
            await websocket.send(response)
    except websockets.ConnectionClosed:
        pass

if __name__ == "__main__":
    # Start the WebSocket server
    start_server = websockets.serve(handle_websocket, "localhost", 12345)

    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

