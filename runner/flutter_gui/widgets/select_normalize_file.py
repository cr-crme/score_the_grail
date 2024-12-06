import os
from pathlib import Path
from typing import Callable

import flet as ft

from ..models.style import title_style


def select_file_widget(page: ft.Page, callback: Callable[[str], None]) -> ft.Control:
    """
    Select a normalized file widget, including the Title, Text, and Icon Button

    Parameters
    ----------
    page : ft.Page
        Page to add the widget to
    callback : Callable[[str], None]
        Callback function to be called when a file is selected

    Returns
    -------
    ft.Control
        The select file widget
    """

    title = ft.Text("Select a normalized file (.csv)", style=title_style)
    selected_file_text = ft.Text(value="No file selected")
    pick_file_dialog = _file_picker_dialog(
        page=page,
        on_result=lambda e: _update_selected_file_text(
            text=selected_file_text, file_path=e.files[0].path, callback=callback
        ),
    )
    initial_folder = Path.cwd() / "data"
    if not initial_folder.exists():
        initial_folder = Path.cwd()

    pick_file_icon_button = ft.IconButton(
        icon=ft.icons.UPLOAD_FILE,
        on_click=lambda _: pick_file_dialog.pick_files(initial_directory=initial_folder, allow_multiple=False),
    )

    return ft.Column(
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        controls=[
            title,
            ft.Row(
                alignment=ft.MainAxisAlignment.CENTER,
                controls=[
                    selected_file_text,
                    pick_file_icon_button,
                ],
            ),
        ],
    )


# Pick files dialog
def _update_selected_file_text(text: ft.Text, file_path: str, callback: Callable[[str], None]) -> None:
    """
    Update the text widget with the new value and call the callback function
    """
    # Remove the Path.cwd() prefix of the file path
    text.value = file_path.replace(f"{Path.cwd()}{os.sep}", "").replace(os.sep, "/")
    text.update()
    callback(file_path)


def _file_picker_dialog(page: ft.Page, on_result: Callable[[ft.FilePickerResultEvent], None]) -> ft.FilePicker:
    """
    Declare and hide the dialog in overlay

    Parameters
    ----------
    page : ft.Page
        Page to add the dialog to
    callback : Callable[[ft.FilePickerResultEvent], None]
        Callback function to be called when the dialog is closed

    Returns
    -------
    ft.Control
        The file picker
    """
    pick_file_dialog = ft.FilePicker(on_result=on_result)
    page.overlay.append(pick_file_dialog)
    return pick_file_dialog
