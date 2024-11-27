import flet as ft

from flutter_gui import select_file_widget


def main(page: ft.Page):

    page.add(
        ft.Column(
            controls=[
                select_file_widget(page, callback=lambda file: print(file)),
            ],
        )
    )


ft.app(target=main)
