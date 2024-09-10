# Customer Churn Prediction Web Application

## Overview

Welcome to the Customer Churn Prediction tool, designed to assist telecom companies in identifying customers who may be at risk of leaving. This web application leverages advanced machine learning techniques to provide actionable insights.

## Features

- User-friendly web interface
- Machine learning model trained on historical customer data
- API for seamless integration with other systems
- Responsive design for optimal viewing on mobile and desktop
- Visualizations to illustrate customer trends
- Robust data security measures

## Technology Stack

- Python
- Flask (web framework)
- Scikit-learn (machine learning)
- Pandas (data manipulation)
- NumPy (numerical computations)
- HTML/CSS (frontend design)
- JavaScript (interactivity)
- Pickle (model serialization)

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/Bakari01/Telco-Customer-Churn.git
   ```

2. Navigate to the project directory:

   ```bash
   cd customer-churn-prediction
   ```

3. Set up a virtual environment:

   ```bash
   python -m venv venv
   ```

4. Activate the virtual environment:
   - For Windows: `venv\Scripts\activate`
   - For Mac and Linux: `source venv/bin/activate`

5. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

6. Create a configuration file:
   Create a file named `.env` and include your environment variables.

7. Start the application:

   ```bash
   python app_new.py
   ```

8. Access the application in your web browser at `http://localhost:5000`.

## Usage Instructions

### Web Interface

1. Visit the home page.
2. Enter customer details in the provided fields.
3. Click the "Predict Churn" button.
4. Review the generated charts and metrics.

### API Access

For developers wishing to integrate the prediction model:

1. Send a POST request to `http://localhost:5000/predict` with the customer data.
2. Receive the prediction response.

## Contributing

We welcome contributions to enhance this project. Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to get involved.

## License

This project is licensed under the MIT License. For more details, please see the [LICENSE](LICENSE) file.

## Acknowledgments

Thank you for exploring the Customer Churn Prediction tool. We appreciate your interest and are here to assist with any questions or feedback.
