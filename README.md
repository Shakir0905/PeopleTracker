# Трекер людей с отображением пройденного пути

Этот проект реализует трекер людей с отображением пройденного пути на видеокадрах. Также включает в себя функции для обнаружения людей с использованием Haar cascades и визуализации трекеров.

## Структура проекта

- `data`: Папка с данными, такими как видео и файлы гомографии.
- `src`: Исходный код проекта.
  - `utils`: Вспомогательные модули.
- `results`: Результаты работы проекта.
  - `bird_eye_views`: Вид сверху с отображением путей.
  - `tracked_videos`: Видео с отображением пройденных путей.

## Запуск проекта

1. Установите необходимые зависимости с помощью `pip install -r requirements.txt`.
2. В папке `data` разместите видео для обработки и файлы гомографии.
3. Запустите `main.py` для обработки видео и трекинга путей.

## Зависимости

- OpenCV
- NumPy
- Matplotlib
