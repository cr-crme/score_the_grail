import flet as ft

from flutter_gui import GenericCallback, select_file_widget, map_and_gps_widget


def main(page: ft.Page):
    callback = GenericCallback()

    page.add(
        ft.Column(
            height=page.height,
            scroll=ft.ScrollMode.ALWAYS,
            controls=[
                select_file_widget(page=page, callback=lambda file_path: callback.notify(file_path)),
                ft.Container(height=20),
                map_and_gps_widget(listener=callback),
            ],
        )
    )


ft.app(target=main)
