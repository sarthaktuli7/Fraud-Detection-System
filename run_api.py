{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "db3e518a-91f7-4ab6-89bb-02e5c5f645c6",
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import os\n",
    "\n",
    "# Add src to path\n",
    "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n",
    "\n",
    "import uvicorn\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    uvicorn.run(\"src.api.fraud_api:app\", host=\"0.0.0.0\", port=8000, reload=True)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.25"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
