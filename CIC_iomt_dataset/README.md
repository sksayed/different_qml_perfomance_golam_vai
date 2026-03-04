This folder is the expected location for the CIC IoMT 2024 dataset used by the quantum models.

Expected files (not tracked in git, see `.gitignore`):

- `CIC_IoMT_2024_WiFi_MQTT_train.parquet` or `CIC_IoMT_2024_WiFi_MQTT_train.csv`
- `CIC_IoMT_2024_WiFi_MQTT_test.parquet` or `CIC_IoMT_2024_WiFi_MQTT_test.csv`

These files are large and are intentionally excluded from version control.  
On a new machine:

1. Clone the repository.
2. Place the dataset files into this `CIC_iomt_dataset` folder.
3. Run the training/evaluation scripts as described in the project README.

