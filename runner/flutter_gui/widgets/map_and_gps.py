import flet as ft
from score_the_grail import KinematicData, NormativeData

from ..models.style import title_style, subtitle_style
from ..models.generic_callback import GenericCallback


def map_and_gps_widget(listener: GenericCallback) -> ft.Control:
    """
    Show the computed Movement Analysis Profile (MAP) and Gait Profile Score (GPS) widget

    Parameters
    ----------
    page : ft.Page
        Page to add the widget to

    Returns
    -------
    ft.Control
        The score widget
    """

    title = ft.Text("The MAP and GPS scores", style=title_style)

    map_score_subtitle = ft.Text(value="MAP", style=title_style, text_align=ft.TextAlign.RIGHT)
    map_score_text_dof_title = ft.Text(value="Segment/Plane", style=subtitle_style, text_align=ft.TextAlign.LEFT)
    map_score_text_left_title = ft.Text(value="Left", style=subtitle_style, text_align=ft.TextAlign.RIGHT)
    map_score_text_right_title = ft.Text(value="Right", style=subtitle_style, text_align=ft.TextAlign.RIGHT)
    map_score_text_dof_data = ft.Text(value="", text_align=ft.TextAlign.LEFT)
    map_score_text_left_data = ft.Text(value="", text_align=ft.TextAlign.RIGHT)
    map_score_text_right_data = ft.Text(value="", text_align=ft.TextAlign.RIGHT)
    map_score_widget = ft.Column(
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        controls=[
            map_score_subtitle,
            ft.Row(
                controls=[
                    ft.Column(
                        horizontal_alignment=ft.CrossAxisAlignment.START,
                        controls=[map_score_text_dof_title, map_score_text_dof_data],
                    ),
                    ft.Column(
                        horizontal_alignment=ft.CrossAxisAlignment.END,
                        controls=[map_score_text_left_title, map_score_text_left_data],
                    ),
                    ft.Column(
                        horizontal_alignment=ft.CrossAxisAlignment.END,
                        controls=[map_score_text_right_title, map_score_text_right_data],
                    ),
                ]
            ),
        ],
    )

    gps_score_subtitle = ft.Text(value="GPS", style=title_style, text_align=ft.TextAlign.RIGHT)
    gps_score_text_dof_title = ft.Text(value="", style=subtitle_style, text_align=ft.TextAlign.LEFT)
    gps_score_text_left_title = ft.Text(value="Left", style=subtitle_style, text_align=ft.TextAlign.RIGHT)
    gps_score_text_right_title = ft.Text(value="Right", style=subtitle_style, text_align=ft.TextAlign.RIGHT)
    gps_score_text_dof_data = ft.Text(value="", text_align=ft.TextAlign.LEFT)
    gps_score_text_left_data = ft.Text(value="", text_align=ft.TextAlign.RIGHT)
    gps_score_text_right_data = ft.Text(value="", text_align=ft.TextAlign.RIGHT)
    gps_score_widget = ft.Column(
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        controls=[
            gps_score_subtitle,
            ft.Row(
                controls=[
                    ft.Column(
                        horizontal_alignment=ft.CrossAxisAlignment.START,
                        controls=[gps_score_text_dof_title, gps_score_text_dof_data],
                    ),
                    ft.Column(
                        horizontal_alignment=ft.CrossAxisAlignment.END,
                        controls=[gps_score_text_left_title, gps_score_text_left_data],
                    ),
                    ft.Column(
                        horizontal_alignment=ft.CrossAxisAlignment.END,
                        controls=[gps_score_text_right_title, gps_score_text_right_data],
                    ),
                ]
            ),
        ],
    )

    listener.listen(
        lambda file_path: _on_file_changed(
            file_path,
            map_score_text_dof_data,
            map_score_text_left_data,
            map_score_text_right_data,
            gps_score_text_dof_data,
            gps_score_text_left_data,
            gps_score_text_right_data,
        )
    )

    return ft.Column(
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        controls=[
            title,
            ft.Row(
                alignment=ft.MainAxisAlignment.CENTER,
                vertical_alignment=ft.CrossAxisAlignment.START,
                controls=[
                    map_score_widget,
                    ft.Container(width=20),
                    gps_score_widget,
                ],
            ),
        ],
    )


def _on_file_changed(
    file_path: str,
    map_score_text_dof: ft.Text,
    map_score_text_left: ft.Text,
    map_score_text_right: ft.Text,
    gps_score_text_dof: ft.Text,
    gps_score_text_left: ft.Text,
    gps_score_text_right: ft.Text,
) -> None:
    # Compute the MAP and GPS scores from the file path
    kd_exported_normalized = KinematicData.from_normalized_csv(file_path)
    map = kd_exported_normalized.map(normative_data=NormativeData.CROUCHGAIT)
    gps = kd_exported_normalized.gps(normative_data=NormativeData.CROUCHGAIT)

    map_dof = ""
    map_value_left = ""
    map_value_right = ""
    for dof, left, right in zip(map.channel_names, map.left.to_numpy, map.right.to_numpy):
        map_dof += f"{dof}\n"
        map_value_left += f"{left:.3f}\n"
        map_value_right += f"{right:.3f}\n"

    map_score_text_dof.value = map_dof
    map_score_text_dof.update()
    map_score_text_left.value = map_value_left
    map_score_text_left.update()
    map_score_text_right.value = map_value_right
    map_score_text_right.update()

    # 3 digits after the decimal point
    gps_score_text_dof.value = f"Total"
    gps_score_text_dof.update()
    gps_score_text_left.value = f"{gps.left.to_numpy:.3f}"
    gps_score_text_left.update()
    gps_score_text_right.value = f"{gps.right.to_numpy:.3f}"
    gps_score_text_right.update()
