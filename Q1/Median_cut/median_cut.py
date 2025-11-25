from PIL import Image

def median_cut_quantize(input_path, output_path, num_colors):
    img = Image.open(input_path).convert("RGB")
    quantized_img = img.quantize(colors=num_colors, method=0)
    quantized_img = quantized_img.convert("RGB")
    quantized_img.save(output_path)
    print("Saved quantized image:", output_path)

num_colors = int(input("Enter No of colors to quantize: "))
median_cut_quantize("input.jpg", "output_median_cut.jpg", num_colors)
