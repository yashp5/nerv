from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib
import uvicorn
from datetime import datetime, timedelta
import pandas as pd
import requests
from io import StringIO
from jose import JWTError, jwt
from passlib.context import CryptContext

app = FastAPI(
    title="Energy Optimization API",
    description="API for optimizing energy usage with demand response.",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to be more restrictive in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# JWT Configuration
SECRET_KEY = "shakalakaboomboom"  # Change this to a secure random string
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# User model
class User(BaseModel):
    username: str
    hashed_password: str


# Token model
class Token(BaseModel):
    access_token: str
    token_type: str


# Dummy user database (replace with a real database in production)
users_db = {
    "testuser": {
        "username": "testuser",
        "hashed_password": pwd_context.hash("testpassword"),
    }
}


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_user(username: str):
    if username in users_db:
        user_dict = users_db[username]
        return User(**user_dict)


def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(username)
    if user is None:
        raise credentials_exception
    return user


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


# Load the model
model = tf.keras.layers.TFSMLayer(
    "/Users/yash/x/ai/model_energy", call_endpoint="serving_default"
)

# Load scalers
scaler_hourly_rates = joblib.load(
    "/Users/yash/x/ai/model_energy/scaler_hourly_rates.pkl"
)
scaler_total_energy = joblib.load(
    "/Users/yash/x/ai/model_energy/scaler_total_energy.pkl"
)
scaler_curtailment = joblib.load("/Users/yash/x/ai/model_energy/scaler_curtailment.pkl")
scaler_real_profiles = joblib.load(
    "/Users/yash/x/ai/model_energy/scaler_real_profiles.pkl"
)


class PredictionRequest(BaseModel):
    hourly_rates: list
    total_energy: float
    curtailment_limit: float

@app.post("/predict/")
def predict(request: PredictionRequest, current_user: User = Depends(get_current_user)):
    # Normalize the input data
    hourly_rates = np.array(request.hourly_rates).reshape(1, -1)
    total_energy = np.array([[request.total_energy]])
    curtailment_limit = np.array([[request.curtailment_limit]])

    hourly_rates_normalized = scaler_hourly_rates.transform(hourly_rates)
    total_energy_normalized = scaler_total_energy.transform(total_energy)
    curtailment_limit_normalized = scaler_curtailment.transform(curtailment_limit)

    input_normalized = np.hstack(
        (hourly_rates_normalized, total_energy_normalized, curtailment_limit_normalized)
    )
    input_reshaped = input_normalized.reshape(
        (input_normalized.shape[0], 1, input_normalized.shape[1])
    )

    # Generate the charging profile
    generated_profile_normalized = model(input_reshaped)

    key = "dense"  # Replace this with the correct key from the printed keys
    generated_profile_tensor = generated_profile_normalized[key]

    # Transform the normalized output to the original scale
    generated_profile = scaler_real_profiles.inverse_transform(
        generated_profile_tensor.numpy()
    )

    def adjust_profile_to_total_energy(profile, total_energy):
        profile_sum = profile.sum()
        if not np.isclose(profile_sum, total_energy):
            adjustment_factor = total_energy / profile_sum
            profile *= adjustment_factor
        return profile

    generated_profile_adjusted = adjust_profile_to_total_energy(
        generated_profile[0], request.total_energy
    )

    return {"generated_profile": generated_profile_adjusted.tolist()}


@app.get("/fetch_pricing_data/")
def fetch_pricing_data(current_user: User = Depends(get_current_user)):
    try:
        now = datetime.now()

        if now.hour >= 10:
            # If the current time is after 10 AM, get tomorrow's date
            target_date = now + timedelta(days=1)
        else:
            # If the current time is before 10 AM, get today's date
            target_date = now

        date_str = target_date.strftime("%Y%m%d")

        # Create the URL with the updated date
        csv_url = f"http://mis.nyiso.com/public/csv/damlbmp/{date_str}damlbmp_zone.csv"

        # Fetch the CSV data
        response = requests.get(csv_url)
        response.raise_for_status()

        # Load the CSV data into a DataFrame
        data = pd.read_csv(StringIO(response.text))

        # Convert 'Time Stamp' column to datetime
        data["Time Stamp"] = pd.to_datetime(data["Time Stamp"])

        # Ensure the numeric columns are converted to numeric types, coercing errors to NaN
        data["LBMP ($/MWHr)"] = pd.to_numeric(data["LBMP ($/MWHr)"], errors="coerce")
        data["Marginal Cost Losses ($/MWHr)"] = pd.to_numeric(
            data["Marginal Cost Losses ($/MWHr)"], errors="coerce"
        )
        data["Marginal Cost Congestion ($/MWHr)"] = pd.to_numeric(
            data["Marginal Cost Congestion ($/MWHr)"], errors="coerce"
        )

        # Filter out rows where Name is not 'N.Y.C.'
        nyc_data = data[data["Name"] == "N.Y.C."].copy()

        # Combine the columns into a single LBMP column
        nyc_data["Total LBMP ($/MWHr)"] = (
            nyc_data["LBMP ($/MWHr)"]
            + nyc_data["Marginal Cost Losses ($/MWHr)"]
            + nyc_data["Marginal Cost Congestion ($/MWHr)"]
        )

        # Drop rows with NaN values in the 'Total LBMP ($/MWHr)' column
        nyc_data = nyc_data.dropna(subset=["Total LBMP ($/MWHr)"])

        # Select only the 'Time Stamp' and 'Total LBMP ($/MWHr)' columns
        nyc_data = nyc_data[["Time Stamp", "Total LBMP ($/MWHr)"]]

        # Create a matrix for the current day's hourly rates
        hourly_rates = (
            nyc_data.set_index("Time Stamp")
            .resample("H")
            .mean()["Total LBMP ($/MWHr)"]
            .values
        )

        # Ensure the hourly rates have 24 samples, fill missing values if necessary
        if len(hourly_rates) < 24:
            hourly_rates = np.pad(
                hourly_rates,
                (0, 24 - len(hourly_rates)),
                "constant",
                constant_values=np.nan,
            )

        # Convert the list to a NumPy array
        hourly_rates = np.array(hourly_rates)

        return {"hourly_rates": hourly_rates.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# To run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
