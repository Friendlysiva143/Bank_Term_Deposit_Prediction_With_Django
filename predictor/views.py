from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd
import os
from joblib import load
import io

def preprocess(df):
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category').cat.codes
    return df

def predict_view(request):
    if request.method == 'POST' and request.FILES.get('csv_file'):
        df = pd.read_csv(request.FILES['csv_file'])

        # Drop label column if exists
        if 'y' in df.columns:
            df = df.drop(columns=['y'])

        df_processed = preprocess(df.copy())

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, 'model/term_deposit_model.joblib')

        if os.path.exists(model_path):
            model = load(model_path)
            preds = model.predict(df_processed)
            preds = ['Yes' if p == 1 else 'No' for p in preds]

            df['Prediction'] = preds  # Append prediction to original dataframe

            # Create CSV in memory
            buffer = io.StringIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)

            # Return CSV response
            response = HttpResponse(buffer, content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename=predictions.csv'
            return response
        else:
            return HttpResponse("Model file not found. Please train the model first.", status=500)

    return render(request, 'predictor/index.html')
