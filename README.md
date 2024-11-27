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
