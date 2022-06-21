import logging

class Log(object):
    def __init__(self, fname = 'log'):
        # cleanup
        logging.shutdown()

        # We log to file and to script
        logging.basicConfig(format='%(asctime)s:%(name)s:%(levelname)s:  %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG,
                            handlers=[logging.FileHandler("log.log"), logging.StreamHandler()])

        logger = logging.getLogger()
        logger.info('Initialized logger')
        self.logger = logger

    def _print(self, msg, level = 'info'):
        try:
            fun = getattr(self.logger, level)
            fun(msg)
        except Exception as e:
            print(msg)


    def __del__(self):
        self.logger.info('Flushing loggers')
        x = logging._handlers.copy()
        for i in x:
        	self.logger.removeHandler(i)
        	i.flush()
        	i.close()
        logging.shutdown() # This should be at the very end
