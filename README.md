# AccidentSeverityBayesianNetwork
python accident_analysis_model.py

Running the main program with –h will bring up a list of optional parameters to enter when running from the command line. The location of the two CSV files can be set with --accident_file and --driver_file, but they are set to the data directory by default. The --checkpoint_dir option can be used if the saved data, model, or prediction files move. These files are saved to be pre-loaded into the model to save time. The --no_dataset_plots flag runs the program without showing the plots in Figure B2. Finally, the --test_program flag can be used to run the program with a very small subset of the data. Doing this will decrease the run time from hours to minutes. When running in test-mode, remove the checkpoint files so they are not pre-loaded into the program.

python test_predictions.py

The test predictions program is used to test the model with a given set of observations. The observations can be set in the main program. The --checkpoint_dir option again shows the path to the saved model and data files. This is defaulted to the checkpoint directory in the repository.
