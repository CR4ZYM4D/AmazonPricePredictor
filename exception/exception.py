import sys

'''
    File to create the exception class 
'''

class ProjectError(Exception):

    def __init__(self, error_message, error_details: sys):
        
        super().__init__()

        self.error_message = error_message
        _, _, self.exception_traceback = error_details.exc_info()

        self.file_name, self.line_num = self.exception_traceback.tb_frame.f_code.co_filename, self.exception_traceback.tb_lineno


    def __str__(self):

        return "Error occured in file [{0}] line number [{1}] message [{2}]".format((self.file_name, self.line_no, str(self.error_message)))    


