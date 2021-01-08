import time
from notion.client import NotionClient

# Obtain the `token_v2` value by inspecting your browser cookies on a logged-in (non-guest) session on Notion.so
logFile = "train.log"
token = "$TOKEN"
pageLink = "$PAGELINK"

client = NotionClient(
    token_v2=token)

while True:
    try:
        page = client.get_block(pagelink)
    except:
        client = NotionClient(
            token_v2=token)
        continue
    content = ""
    with open(logFile, "r") as f:
        content = f.read()
    print("System time %s, upload log %s to Notion. Sleep for 900s." %
          (time.asctime(time.localtime(time.time())), logFile))
    page.children[0].title = content
    time.sleep(900)
