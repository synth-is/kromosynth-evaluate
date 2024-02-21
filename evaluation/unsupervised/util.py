import hashlib

# Function to convert a file path to a unique-ish port number between 1024 and 65535
# - there can be collisions, due to the pigeonhole principle
# todo
def filepath_to_port(filepath):
    hash_val = hashlib.md5(filepath.encode('utf-8')).hexdigest()
    short_hash = int(hash_val[:8], 16)  # Take first 8 characters for a wider range of values
    port = 1024 + short_hash % (65535 - 1024)  # Add 1024 to ensure port is above 1023
    return port

# Alternatively check if the port is already in use and try another one - not tested:
# import hashlib
# import socket

# def bind_to_port(port):
#     # Create a new socket
#     sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     # Try to bind to the port
#     try:
#         sock.bind(("", port))  # We don't specify a host, so it defaults to localhost
#         sock.close()  # Close the socket, we just wanted to check if we could open it
#         return True
#     except OSError:  # The OS raises an OSError if the port is in use
#         return False

# def filepath_to_port(filepath):
#     filepath_encoded = filepath.encode("utf-8")
#     filepath_length = len(filepath_encoded)
    
#     # the filepath_length variable is used to generate different hash values
#     # until an unused port is found
#     used_filepath_length = filepath_length

#     while True:
#         variation = (used_filepath_length % filepath_length) + 1
#         hashed_filepath = hashlib.md5(filepath_encoded * variation).hexdigest()
#         short_hash = int(hashed_filepath[:8], 16)
#         port = 1024 + short_hash % (65535 - 1024)

#         # If the port isn't in use
#         if bind_to_port(port):
#             return port  # Return the port
#         else:
#             # If it is in use, increase used_filepath_length and try again
#             used_filepath_length += 1

# # Example usage:
# filepath = "/home/user/directory/file.txt"
# port_number = filepath_to_port(filepath)
# print(port_number)