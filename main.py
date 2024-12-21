import time

from semantic_router import Route
from semantic_router.layer import RouteLayer
from dotenv import load_dotenv
from semantic_router.encoders import OpenAIEncoder
# from semantic_router.encoders import CohereEncoder

load_dotenv()

begin = time.time()

# we could use this as a guide for our chatbot to avoid political conversations
politics = Route(
    name="politics",
    utterances=[
        "isn't politics the best thing ever",
        "why don't you tell me about your political opinions",
        "don't you just love the president",
        "they're going to destroy this country!",
        "they will save the country!",
    ],
)

# this could be used as an indicator to our chatbot to switch to a more
# conversational prompt
chitchat = Route(
    name="chitchat",
    utterances=[
        "how's the weather today?",
        "how are things going?",
        "lovely weather today",
        "the weather is horrendous",
        "let's go to the chippy",
    ],
)

# we place both of our decisions together into single list
routes = [politics, chitchat]

encoder = OpenAIEncoder()

# With our routes and encoder defined we now create a RouteLayer. The route layer handles our semantic decision making.
rl = RouteLayer(encoder=encoder, routes=routes)

routes_output = rl("don't you love politics?").name

print(f"Time taken: {time.time() - begin}")

print(routes_output)
