import json
from logging import raiseExceptions
import re
import traceback


class node(object):
    def __init__(self, fa=None, token="", type_="") -> None:
        super().__init__()
        self.fa = fa
        self.child = []
        self.token = token
        self.type = type_
        self.answer = None
        self.real_program = ""
    def getfa(self):
        return self.fa
    def getchild(self, index):
        return self.child[index]

eps = 1e-6
class executor(object):
    def __init__(self, program, num_dict)->None:
        super().__init__()
        if program == None:
            program = ""
        self.program = program
        self.number2token = num_dict

        self.rt = None

    def tokenize(self, program: str):
        if program.strip() == "":
            raise Exception("No tokenized program!")

        _type = [] #str or number or operator
        _token = []


        def get_word(program , idx): #for number tokens in program
            fl = idx
            while fl < len(program):
                if program[fl] == "]":
                    break
                fl += 1
            return program[idx:fl+1], fl+1


        idx = 0
        while idx < len(program):
            s = program[idx]
            if s == " ":
                idx += 1
                continue
            elif s == '(':
                _token.append(s)
                _type.append('leftpare')
                idx += 1
                continue
            elif s == ')':
                _token.append(s)
                _type.append('rightpare')
                idx += 1
                continue
            elif s == ',':
                _token.append(s)
                _type.append('comma')
                idx += 1
                continue
            elif s == '[':
                word,t = get_word(program, idx)
                _token.append(word)
                _type.append('number')
                idx = t
                continue
            elif s in ["+","-","*","/","=","<",">","&","|","!"]:
                _token.append(s)
                _type.append('operator')
                idx += 1
                continue
            else :
                raise Exception(f"Unknown Operator! String is {s}")

        return {'token':_token,'type':_type}

    def get_program_syntax_tree(self):
        if self.rt :
            return
        program = self.program
        if isinstance(program, str):
            program = self.tokenize(program)
        else :
            raise Exception("Program is not even a string!")
        _token = program['token']
        _type = program['type']
        now = node(token="ROOT",type_="ROOT")
        self.rt = now

        pre = ""
        for i in range(len(_token)):

            token= _token[i]
            type_ = _type[i]
            if type_ == 'operator':
                temp = node(fa=now,token=token,type_=type_)
                now.child.append(temp)
                now = temp
            elif type_ == "number":
                temp = node(fa=now,token=token,type_=type_)
                now.child.append(temp)
            elif type_ == 'comma':
                pass
            elif type_ == "leftpare":
                pass
            elif type_ == "rightpare":
                now = now.fa
            pre = type_
        if self.rt is not now:
            raise Exception("Error Expression!")

    def compute_answer(self, n):
        if n.answer != None:
            return
        for c in n.child:
            self.compute_answer(c)
        if n.type == 'operator':
            if n.token == "+":
                n.answer = 0
                for c in n.child:
                    n.answer += c.answer
            elif n.token == "-":
                if len(n.child) == 2:
                    n.answer = n.child[0].answer - n.child[1].answer
                elif len(n.child) == 1:
                    n.answer = (- n.child[0].answer)
                else :
                    raise Exception("- is used incorrectly!")
            elif n.token == "*":
                n.answer = n.child[0].answer * n.child[1].answer
            elif n.token == "/":
                n.answer = n.child[0].answer / n.child[1].answer
            elif n.token == "=":
                n.answer =  (abs(n.child[0].answer - n.child[1].answer) < eps )
            elif n.token == ">":
                n.answer = (n.child[0].answer > n.child[1].answer)
            elif n.token == "<":
                n.answer = (n.child[0].answer < n.child[1].answer)
            elif n.token == "!":
                n.answer = (not n.child[0].answer)
            elif n.token == "&":
                n.answer = (n.child[0].answer & n.child[1].answer)
            elif n.token == "|":
                n.answer = (n.child[0].answer | n.child[1].answer)
            else :
                raise Exception(f"Unknown Operator:{n.token}!")
        elif n.type == 'number':
            n.answer = self.number2token[n.token]
            n.answer = float(n.answer)
        elif n.type == 'ROOT':
            n.answer = n.child[0].answer

        #print(f"answer:{n.answer} token:{n.token} ")
    def get_answer(self):
        try:
            self.get_program_syntax_tree()
            self.compute_answer(self.rt)
            return (True,self.rt.answer)
        except Exception as e :
            errormsg = self.program+" can't run program!"+" "+str(traceback.format_exc())
            return (False,errormsg)

    def transfer_program(self, n):
        if n.real_program != "":
            return
        for c in n.child:
            self.transfer_program(c)
        if n.type == 'operator':
            if n.token == "+":
                n.real_program = "add("
                for c in n.child:
                    n.real_program += c.real_program + ","
                n.real_program = n.real_program[:-1] + ")"
            elif n.token == "-":
                if len(n.child) == 2:
                    n.real_program = "sub(" + n.child[0].real_program + "," + n.child[1].real_program + ")"
                elif len(n.child) == 1:
                    n.real_program = "sub(" + n.child[1].real_program + ")"
                else :
                    raise Exception("- is used incorrectly!")
            elif n.token == "*":
                n.real_program = "mul(" + n.child[0].real_program + "," + n.child[1].real_program + ")"

            elif n.token == "/":
                n.real_program = "div(" + n.child[0].real_program + "," + n.child[1].real_program + ")"
            elif n.token == "=":
                n.real_program =  n.child[0].real_program + " = " + n.child[1].real_program
            elif n.token == ">":
                n.real_program =  n.child[0].real_program + " > " + n.child[1].real_program
            elif n.token == "<":
                n.real_program =  n.child[0].real_program + " < " + n.child[1].real_program
            elif n.token == "!":
                n.real_program = "!" + n.child[0].real_program
            elif n.token == "&":
                n.real_program =  n.child[0].real_program + " & " + n.child[1].real_program
            elif n.token == "|":
                n.real_program =  n.child[0].real_program + " | " + n.child[1].real_program
            else :
                raise Exception(f"Unknown Operator:{n.token}!")
        elif n.type == 'number':
            n.real_program = n.token
        elif n.type == 'ROOT':
            n.real_program = n.child[0].real_program


    def get_transfer_program(self):
        try:
            self.get_program_syntax_tree()
            self.transfer_program(self.rt)
            return (True,self.rt.real_program)
        except Exception as e :
            errormsg = self.program+" can't get real program!"+" "+str(e)
            return (False,errormsg)

class executor_new(object):
    def __init__(self, program, num_dict)->None:
        super().__init__()
        if program == None:
            program = ""
        self.program = program
        self.number2token = num_dict

        self.rt = None

    def tokenize(self, program: str):
        if program.strip() == "":
            raise Exception("No tokenized program!")

        _type = [] #str or number or operator
        _token = []


        def get_word(program , idx): #for number tokens in program
            fl = idx
            while fl < len(program):
                if program[fl] == "]":
                    break
                fl += 1
            return program[idx:fl+1], fl+1
        def get_operator(string, start_idx):
            idx = start_idx
            while idx < len(string):
                s = string[idx]
                if (s > 'z' or s < 'a') and (s > 'Z' or s < 'A'):
                    break
                idx += 1
            return string[start_idx: idx], idx



        idx = 0
        while idx < len(program):
            s = program[idx]
            if s == " ":
                idx += 1
                continue
            elif s == '(':
                _token.append(s)
                _type.append('leftpare')
                idx += 1
                continue
            elif s == ')':
                _token.append(s)
                _type.append('rightpare')
                idx += 1
                continue
            elif s == ',':
                _token.append(s)
                _type.append('comma')
                idx += 1
                continue
            elif s == '[':
                word,t = get_word(program, idx)
                _token.append(word)
                _type.append('number')
                idx = t
                continue
            elif s in ["=","<",">","&","|","!"]:
                if program[idx:idx+2] == "!=":
                    s = program[idx:idx+2]
                    idx += 1

                _token.append(s)
                _type.append('operator')
                idx += 1
                continue
            else :
                word, t = get_operator(program, idx)
                if word not in ["add","sub","div","mul"]:
                    raise Exception(f"Unknown Operator! String is {s}")
                _token.append(word)
                _type.append("operator")
                idx = t

        return {'token':_token,'type':_type}

    def build_syntax_tree(self, now, _token, _type):

        # & |
        #print(_token)
        #print(_type)
        and_or_flags = []
        for i in range(len(_token)):

            token = _token[i]
            type_ = _type[i]
            if type_ == 'operator' and token in ['&','|']:
                and_or_flags.append(i)

        if len(and_or_flags) > 0:
            index = and_or_flags[-1]
            token = _token[index]
            type_ = _type[index]
            temp = node(fa=now,token=token,type_=type_)
            #print(token)
            now.child.append(temp)
            self.build_syntax_tree(temp, _token[:index], _type[:index])
            self.build_syntax_tree(temp, _token[index+1:], _type[index+1:])
            return
        # !=
        not_equal_flags = []
        for i in range(len(_token)):

            token = _token[i]
            type_ = _type[i]
            if type_ == 'operator' and token in ['!=']:
                not_equal_flags.append(i)

        if len(not_equal_flags) > 0:
            index = not_equal_flags[-1]
            token = _token[index]
            type_ = _type[index]
            temp = node(fa=now,token=token,type_=type_)
            #print(token)
            now.child.append(temp)
            self.build_syntax_tree(temp, _token[:index], _type[:index])
            self.build_syntax_tree(temp, _token[index+1:], _type[index+1:])
            return

        # = > <
        equal_greater_smaller_flags = []
        for i in range(len(_token)):

            token = _token[i]
            type_ = _type[i]
            if type_ == 'operator' and token in ['=','<',">"]:
                equal_greater_smaller_flags.append(i)

        if len(equal_greater_smaller_flags) > 0:
            index = equal_greater_smaller_flags[-1]
            token = _token[index]
            type_ = _type[index]
            temp = node(fa=now,token=token,type_=type_)
            #print(token)
            now.child.append(temp)
            self.build_syntax_tree(temp, _token[:index], _type[:index])
            self.build_syntax_tree(temp, _token[index+1:], _type[index+1:])
            return

        # !
        noequal_flags = []
        for i in range(len(_token)):

            token = _token[i]
            type_ = _type[i]
            if type_ == 'operator' and token in ['!']:
                noequal_flags.append(i)

        if len(noequal_flags) > 0:
            index = noequal_flags[-1]
            token = _token[index]
            type_ = _type[index]
            temp = node(fa=now,token=token,type_=type_)
            #print(token)
            now.child.append(temp)
            self.build_syntax_tree(temp, _token[index+1:], _type[index+1:])
            return
        # normal

        for i in range(len(_token)):

            token= _token[i]
            type_ = _type[i]
            if type_ == 'operator':
                temp = node(fa=now,token=token,type_=type_)
                now.child.append(temp)
            elif type_ == "number":
                temp = node(fa=now,token=token,type_=type_)
                now.child.append(temp)
            elif type_ == 'comma':
                pass
            elif type_ == "leftpare":
                now = now.child[-1]
            elif type_ == "rightpare":
                now = now.fa


    def get_program_syntax_tree(self):
        if self.rt :
            return
        program = self.program
        if isinstance(program, str):
            program = self.tokenize(program)
        else :
            raise Exception("Program is not even a string!")
        _token = program['token']
        _type = program['type']
        now = node(token="ROOT",type_="ROOT")
        self.rt = now

        pre = ""
        self.build_syntax_tree(self.rt, _token, _type)



    def compute_answer(self, n):
        if n.answer != None:
            return

        for c in n.child:
            self.compute_answer(c)
        #print(1)



        if n.type == 'operator':
            if n.token == "add":
                n.answer = 0
                for c in n.child:
                    n.answer += c.answer
            elif n.token == "sub":
                if len(n.child) == 2:
                    n.answer = n.child[0].answer - n.child[1].answer
                elif len(n.child) == 1:
                    n.answer = (- n.child[0].answer)
                else :
                    raise Exception("- is used incorrectly!")
            elif n.token == "mul":
                n.answer = n.child[0].answer * n.child[1].answer
            elif n.token == "div":
                n.answer = n.child[0].answer / n.child[1].answer
            elif n.token == "=":
                n.answer =  (abs(n.child[0].answer - n.child[1].answer) < eps )
            elif n.token == "!=":
                n.answer =  (abs(n.child[0].answer - n.child[1].answer) >= eps )
            elif n.token == ">":
                n.answer = (n.child[0].answer > n.child[1].answer)
            elif n.token == "<":
                n.answer = (n.child[0].answer < n.child[1].answer)
            elif n.token == "!":
                n.answer = (not n.child[0].answer)
            elif n.token == "&":
                n.answer = (n.child[0].answer & n.child[1].answer)
            elif n.token == "|":
                n.answer = (n.child[0].answer | n.child[1].answer)
            else :
                raise Exception(f"Unknown Operator:{n.token}!")
        elif n.type == 'number':
            n.answer = self.number2token[n.token]
            n.answer = float(n.answer['v'])
        elif n.type == 'ROOT':
            n.answer = n.child[0].answer

    def get_answer(self):
        try:
            self.get_program_syntax_tree()
            self.compute_answer(self.rt)
            return (True,self.rt.answer)
        except Exception as e :
            errormsg = self.program+" can't run program!"+" "+str(traceback.format_exc())
            return (False,errormsg)


def nli_label_decider(eformula, cformula, num_dict):

    eifrun, e_value = executor_new(eformula, num_dict).get_answer()
    if eifrun is False:
        e_value = False
    cifrun, c_value = executor_new(cformula, num_dict).get_answer()
    if cifrun is False:
        c_value = False

    elif cformula == '!ENTAIL':
        c_value = not e_value
    else:
        c_value = None

    if e_value is None and c_value is None:
        return "undecidable"

    if (e_value == True) and (c_value != True):
        return "entailment"

    if (e_value != True) and (c_value == True):
        return "contradiction"

    if (e_value != True) and (c_value != True):
        return "neutral"
