from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd
import os
from joblib import load
import io

# Columns used for one-hot encoding
CAT_COLUMNS = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
BOOL_COLUMNS = ['housing', 'loan']

def preprocess(df, model_columns):
    # One-hot encode categorical features
    for col in CAT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna('missing')
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(col, axis=1), dummies], axis=1)

    # Convert 'yes'/'no' to binary for boolean columns
    for col in BOOL_COLUMNS:
        if col in df.columns:
            df[col + '_new'] = df[col].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
            df.drop(col, axis=1, inplace=True)

    # Ensure all expected model columns are present
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0  # add missing columns with default value

    # Remove extra columns not used by the model
    df = df[model_columns]

    return df

def predict_view(request):
    if request.method == 'POST' and request.FILES.get('csv_file'):
        df = pd.read_csv(request.FILES['csv_file'])

        # Drop label column if exists
        if 'y' in df.columns:
            df = df.drop(columns=['y'])

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, 'model/term_deposit_prediction_model.joblib')

        if os.path.exists(model_path):
            model = load(model_path)
            model_columns = model.feature_names_in_  # Needed to align features

            df_processed = preprocess(df.copy(), model_columns)

            preds = model.predict(df_processed)
            preds = ['Yes' if p == 1 else 'No' for p in preds]

            df['Prediction'] = preds  # Append predictions to original

            buffer = io.StringIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)

            response = HttpResponse(buffer, content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename=predictions.csv'
            return response
        else:
            return HttpResponse("Model file not found. Please train the model first.", status=500)

    return render(request, 'predictor/index.html')
