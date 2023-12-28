import logging


def safe_init_jvm():
    try:
        from weka.core import jvm

        jvm.start(packages=True, system_cp=True)
    except ImportError:
        logging.warning("Weka not installed. Skipping JVM initialization.")


def safe_stop_jvm():
    try:
        from weka.core import jvm

        jvm.stop()
    except ImportError:
        logging.warning("Weka not installed. Skipping JVM shutdown.")
