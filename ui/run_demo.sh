#!/usr/bin/env bash
# ui/run_demo.sh — start/stop/restart/status para la demo de Gradio (ui/app.py).
#
# Corre en el servidor donde vive el .venv del proyecto (no localmente).
# Uso:
#   ssh server-casa "cd ~/Proyecto/plate-vision-pipeline && ./ui/run_demo.sh start"
#   ssh server-casa "cd ~/Proyecto/plate-vision-pipeline && ./ui/run_demo.sh stop"
#   ssh server-casa "cd ~/Proyecto/plate-vision-pipeline && ./ui/run_demo.sh restart"
#   ssh server-casa "cd ~/Proyecto/plate-vision-pipeline && ./ui/run_demo.sh status"
#
# stop() mata por PUERTO (fuser -k), no por nombre de proceso (pkill -f) —
# un pkill -f 'ui/app.py' se auto-matchea con la línea de comando del propio
# SSH que lo invoca y mata la sesión antes de hacer nada.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
LOG_FILE="/tmp/gradio_demo.log"
PORT=7860

is_up() {
    curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/" --max-time 2 | grep -q 200
}

status() {
    if is_up; then
        echo "Demo corriendo — http://127.0.0.1:${PORT}"
    else
        echo "Demo NO está corriendo."
    fi
}

start() {
    if is_up; then
        echo "Ya está corriendo en el puerto ${PORT}."
        return 0
    fi
    cd "$PROJECT_DIR"
    rm -f "$LOG_FILE"
    setsid nohup "$VENV_PYTHON" ui/app.py > "$LOG_FILE" 2>&1 < /dev/null &
    disown
    sleep 3
    status
}

stop() {
    if fuser -k "${PORT}/tcp" 2>/dev/null; then
        echo "Detenida."
    else
        echo "Nada escuchando en el puerto ${PORT}."
    fi
}

restart() {
    stop
    sleep 1
    start
}

case "${1:-}" in
    start) start ;;
    stop) stop ;;
    restart) restart ;;
    status) status ;;
    *)
        echo "Uso: $0 {start|stop|restart|status}" >&2
        exit 1
        ;;
esac
