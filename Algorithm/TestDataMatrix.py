from pylibdmtx.pylibdmtx import encode
from PIL import Image

text = "W980680107523A100"
encoded = encode(text.encode('utf-8'))
img = Image.frombytes('L', (encoded.width, encoded.height), encoded.pixels)
img.save("datamatrix.png")
img.show()