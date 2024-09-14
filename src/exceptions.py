import sys

def display_error_message(error,error_detail:sys):

    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error= str(error)
    error_message = f'Error occured at python script {file_name} at line number {line_number} message {error}'
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=display_error_message(error_message,error_detail)

    def __str__(self):
        return  self.error_message    