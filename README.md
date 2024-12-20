# score_the_grail
A simple GUI application to compute walking score (MAP and GPS) using the GRAIL

## GUI
    To run the GUI, you will need to install the `flet` package which is automatically install when using `pip install .`.
    Alternatively, you can manually install it using the following command:
    ```bash
    pip install flet
    ```

    On Linux, you may need to add the `mpv` and `zenity` libraries. 
    You can install then using the following commands:
    ```bash
    sudo apt install libmpv-dev libmpv2
    sudo ln -s /usr/lib/x86_64-linux-gnu/libmpv.so /usr/lib/libmpv.so.1
    sudo apt install zenity
    ```
## Generating the executable

    You will first need to install all the dependencies and the compiler, using the following command:
    ```bash
    cd $ROOT
    pip install .
    pip install pyinstaller
    ```

    Then navigate to the `$ROOT/runner` folder and run the `gui_to_executable.py` script
    ```bash
    cd $ROOT/runner
    python gui_to_executable.py
    ```

    The executable will be generated in the `$ROOT/runner/dist` folder.

## Normative data

The normative data were happily provided by the GRAIL team. 
They are stored in the `data` folder as encrypted files.
To decrypt the data, one must set the `NORMATIVE_GRAIL_DATA_KEY` environment variable to the key provided. 
To get a key, please contact us at `pariterre@hotmail.com`.

