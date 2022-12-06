import streamlit as st

# Load the model and create predictions
@st.cache(allow_output_mutation=True)
def predict(inputs):
    # Load the model and make predictions
    model = load_model('model.h5')
    predictions = model.predict(inputs)

    return predictions

# Create the main app
def main():
    st.title("ML Model Deployment with Streamlit")

    # Get user inputs
    inputs = st.text_input("Enter input values:")

    # Make predictions and display the results
    if st.button("Predict"):
        predictions = predict(inputs)
        st.write(predictions)

# Run the app
if __name__ == '__main__':
    main()
