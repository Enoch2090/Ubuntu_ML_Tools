import time
from notion.client import NotionClient

# Obtain the `token` value by inspecting your browser cookies on a logged-in (non-guest) session on Notion.so for `token_v2`.
# Settings:
logFile = "train.log"
token = "$TOKEN"
pageLink = "$PAGELINK"
period = 900
lines = 50

client = NotionClient(
    token_v2=token)
while True:
    try:
        page = client.get_block(pageLink)
    except:
        client = NotionClient(
            token_v2=token)
        continue
    content = []
    with open(logFile, "r") as f:
        content = f.readlines()
    print("System time %s, upload log %s to Notion. Sleep for 900s." %
          (time.asctime(time.localtime(time.time())), logFile))
    page.children[0].title = "Updated at %s \n\n %s" % (
        time.asctime(time.localtime(time.time())), "".join(content[-lines::]))
    time.sleep(period)
