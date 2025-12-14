## AI Summary

A file named tk.py.


### Function: inputhook(inputhook_context)

**Description:** Inputhook for Tk.
Run the Tk eventloop until prompt-toolkit needs to process the next input.

### Function: wait_using_filehandler()

**Description:** Run the TK eventloop until the file handler that we got from the
inputhook becomes readable.

### Function: wait_using_polling()

**Description:** Windows TK doesn't support 'createfilehandler'.
So, run the TK eventloop and poll until input is ready.

### Function: done()
