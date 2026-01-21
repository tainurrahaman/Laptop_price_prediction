import pandas as pd
import pickle
import gradio as gr

#load saving model
with open("laptop_price_rf_model.pkl", "rb") as file:
    rf_model = pickle.load(file)

# Main logic
def predictLaptopPrice(brand,name,spec_rating,processor,CPU,Ram,Ram_type,ROM,ROM_type,GPU,display_size, resolution_width,resolution_height,OS,warranty):

    input_df = pd.DataFrame([[brand,name,spec_rating,processor,CPU,Ram,Ram_type,ROM,ROM_type,GPU,display_size,resolution_width,resolution_height,OS,warranty]],
                            columns=['brand','name','spec_rating','processor','CPU','Ram','Ram_type','ROM','ROM_type','GPU','display_size','resolution_width','resolution_height','OS','warranty'])
    
    predictions = rf_model.predict(input_df)[0]

    return f"The predicted price is: {predictions:.2f}"

inputs = [
    gr.Dropdown(["HP","Acer","Lenovo","Apple"],label="brand"),
    gr.Dropdown(["Victus 15-fb0157AX Gaming Laptop","15s-fq5007TU Laptop","One 14 Z8-415 Laptop","Yoga Slim 6 14IAP8 82WU0095IN Laptop","MacBook Air 2020 MGND3HN Laptop"],label="name"),
    gr.Number(label="spec_rating", value=70.0),
    gr.Dropdown(["5th Gen AMD Ryzen 5 5600H","12th Gen Intel Core i3 1215U","11th Gen Intel Core i3 1115G4","12th Gen Intel Core i5 1240P","Apple M1"], label="Processor"),
    gr.Dropdown(["Hexa Core, 12 Threads","Hexa Core (2P + 4E), 8 Threads","Dual Core, 4 Threads","12 Cores (4P + 8E), 16 Threads","Octa Core (4P + 4E)"],label="CPU"),
    gr.Number(label="RAM", value=8),
    gr.Dropdown(["DDR4","LPDDR5"],label="Ram_type"),
    gr.Number(label="ROM", value=512),
    gr.Dropdown(["SSD","HDD"],label="Rom_type"),
    gr.Dropdown(["4GB AMD Radeon RX 6500M","Intel UHD Graphics","Intel Iris Xe Graphics","Intel Integrated Iris Xe","Apple M1 Integrated Graphics"],label="GPU"),
    gr.Number(label="display_size", value=15),
    gr.Number(label="resolution_width", value=2000),
    gr.Number(label="resolution_height", value=1400),
    gr.Dropdown(["Windows 11 OS","Mac OS","Windows 10 OS"],label="OS"),
    gr.Number(label="warranty", value=1),

]
# Interface
app = gr.Interface(
    fn = predictLaptopPrice,
    inputs=inputs,
    outputs="text",
    title="Laptop Price Predictor"
)

# Launch
app.launch()