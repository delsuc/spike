#!/usr/bin/env python
# encoding: utf-8
"""
ConfigForm.py

Handles ConfigParser files via a html interface

Creates a HTML form from the cfg file and store a modified file

uses :

Flask
wtforms
jinja2


Entry in the config files should of the form :


# comment on section1
# comment on section1
[section1]

# comment on key1
# comment on key1
# %tag%
key1 = value

# comment on key2
# comment on key2
# %tag% %validators%
key2 = value

Remark 
valid %tag% are    and %validators% (optionnal except for %options% )

'%boolean%':            -
'%text%':           %min% %max% %extension% 
'%float%':          %min% %max% 
'%integer%':        %min% %max% 
'%radio%':          %options%
'%select%':         %options%
'%file%':           %extension%
'%infile%':         %extension%
'%outfile%':        %extension%
'%hidden%':

valid %validators% are
%min:val%    %max:val%   : limit on the value or on the length of the string
%options:list:of:valid:values%  : list of options
%extension:.txt% : constraint the entry to finish by a given string

Created by DELSUC Marc-Andre on 2015-04-15.
"""
# Still to be done
# BUGS
# - doc of first section
# - booleans
# - debug %extension    and make it multiple
# - two options with same name in 2 diff sections collide !
# - reload() does not recompute template ??
# improvments
# - move ConfigForm.opt_templ to cfField
# - construct an OrderedDict of cfField's in ConfigForm and use it everywhere
# - better layout / css
# - find a way to stop the app
# - use flash message for validation

import sys
import os
import unittest
from collections import OrderedDict, defaultdict
import ConfigParser
import threading, webbrowser
import random
import socket

import wtforms
from wtforms import Form, validators, ValidationError
from jinja2 import Template
from flask import Flask, Response, render_template, request, url_for, redirect

###########
HOST = "localhost"      # the name of the web service, if localhost, the service is only local
PORT = 5000             # the port of the service - choose a free one
Debug = False
StartWeb = True         # If true, opens a browser when launched
__version__ = "0.2"     # 0.1 first trial 0.2 performs ok but still rough
###########
#mini templates
head = """
<html>
<head>
<title>{{title}}</title>
</head>
<body>
<h1>Configuration for file {{filename}}</h1>
"""

foot = """
<p><small><i>author : M.A. Delsuc</i></small></p>
</body>
</html>
"""

# known Field types syntax {%key%: (Field,type)}
cfFieldDef = {
    '%boolean%': (wtforms.BooleanField, bool),
    '%text%': (wtforms.StringField, str),
    '%float%':   (wtforms.FloatField, float),
    '%integer%':   (wtforms.IntegerField, int),
    '%radio%':   (wtforms.RadioField, str),
    '%file%':   (wtforms.FileField, str),
    '%infile%':   (wtforms.FileField, str),
    '%outfile%':   (wtforms.StringField, str),
    '%select%':   (wtforms.SelectField, str),
    '%hidden%':  (wtforms.HiddenField, str)
}
cfFieldKeys = cfFieldDef.keys()
if Debug: print cfFieldKeys

# known validators syntax {%key: Validator}  # Validator field not used so-far
cfvalDef = {
    "%options": validators.AnyOf,
    "%extension":  validators.Regexp,
    "%min": validators.NumberRange,
    "%max": validators.NumberRange
}
cfvalDefKeys = cfvalDef.keys()
if Debug: print cfvalDefKeys

############

def dForm(dico):
    """
    This function creates a wtforms object from the description given in the (Ordered) dict dico 
    dico = {"name_of_field" : Field(), ...}

    This is the key, wtforms requires a class describing the form, we have to build it dynamicaly.
    """
    class C(Form):
        pass
    for i,j in dico.items():
        setattr(C,i,j)
    return C
def dynatemplate(kw, method, action, class_):
    "builds a simplistic template string, see  dTemplate()  - used for tests -"
    temp = ['<form method="{0}" action="{1}">'.format(method,action)]
    for i in kw:
        temp.append('<div>{{{{ form.{0}.label }}}}: {{{{ form.{0}(class="{1}") }}}}</div>'.format(i,class_) )
    temp.append('</form>')
    return "\n".join(temp)
def dTemplate(dico, method, action, class_):
    """
    This function creates a simple Template object from the description given in the (Ordered) dict dico
    method is either POST or GET
    action is the callback url
    class_ is the css class attached to each field
    
    used for tests
    """
    return Template(dynatemplate(dico, method, action, class_))


class cfField(object):
    """
    This class holds all the parameters of an option in the config file

    subparse_comment() loads it    
    """
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.type_ =  "%text%"
        self.doc = ""
        self.section = ""
        self.lopt = None
        self.meta = None      # meta will be used by produce()
        self.validators = [validators.DataRequired()]   # all field are required, initiailize the list
    def subparse_comment(self, line):
        """parse the comment, and search for meta comments"""
        words = line.split()
        word = words.pop(0)     # get first one
        if word not in cfFieldKeys:  # we do not have a meta comment
            self.doc.append(line)
            return
        self.type_ = word       # now we do !
        self.meta = line
        for word in words:      # then process meta
            val = word[:-1].split(":")  # extract sub options, removing the final %
            if val[0] in cfvalDefKeys:  # we have a validator
                if val[0] == "%min":
                    if cfFieldDef[self.type_][1] == str:     # min and max are both numbers and string
                        self.validators.append( validators.Length(min=int(val[1])) )
                    else:
                        self.validators.append( validators.NumberRange(min=float(val[1])) )
                elif val[0] == "%max":
                    if cfFieldDef[self.type_][1] == str:     # min and max are both numbers and string
                        self.validators.append( validators.Length(max=int(val[1])) )
                    else:
                        self.validators.append( validators.NumberRange(max=float(val[1])) )
                elif val[0] == "%extension":
                    print "%Etension:xxx% is currently not implemented"
                    print "XX EXT", val[0], '%s$'%(val[1],)
#                    self.validators.append( validators.Regexp(r'%s$'%val[1], message="should end with %s"%val[1]) )
                elif  val[0] == "%options":
                    lopt = val[1:]
                    self.lopt = lopt
                    self.validators.append( validators.AnyOf(values=lopt ) )
    def __repr__(self):
        return "%s_%s %s %s %s\n%s"%(self.section, self.name, self.value, self.type_, self.validators, self.doc)
    def __str__(self):
        return self.__repr__()
class ConfigForm(object):
    """
    This class reads a configparser file
    creates a HTML form for it
    starts a broswer on it (using Flask) which allows to modify the values and store them back.

    """
    def __init__(self, cffilename):
        self.cffilename = cffilename
        self.cp = ConfigParser.ConfigParser()
        self.configfile = open(cffilename).readlines()
        self.cp.read(cffilename)
        self.doc_sec = {}   # contains comments for sections
        self.doc_opt = {}   # and options
        self.options = OrderedDict()   # contains all cfFormField()
        self.read_comments()
        self.parse_comments()
        self.method = 'POST'
        self.callback = '/callback'
        self.class_ = "CF"      # this will be used in html templates
        self.port = PORT 
        self.host = HOST
        self.template = None
        self.form = None
    def reload(self):
        "reload the content from the file"
        self.configfile = open(self.cffilename).readlines()
        self.cp.read(self.cffilename)
        self.template = None        # clear cache
        self.form = None            # clear cache
    def read_comments(self):
        """
        this method reads the comments associated to sections and options
        comments should be put BEFORE the commented entry
        ## comments (double #) are silently skipped
        """
        currentsection = "initial"
        currentoption = None
        currentcom = []
        doc_sec = defaultdict(list)
        doc_opt = defaultdict(list)
        for l in self.configfile:
            l.strip()
            if l.startswith('##'):    # skip double comments
                continue
            if l.startswith('#'):     # store comments
                currentcom.append(l[1:].strip())
                continue
            m = self.cp.SECTCRE.match(l)
            if m:  # a new section
                currentsection = m.group('header').strip().lower()  # ConfigParser puts all keys in lower case
                currentoption = None
                doc_sec[currentsection] += currentcom
                currentcom = []
            m = self.cp.OPTCRE.match(l)
            if m:  # a new option
                currentoption = m.group('option').strip().lower()
                doc_opt[currentoption] += currentcom
                currentcom = []
        self.doc_sec = doc_sec
        self.doc_opt = doc_opt
    def parse_comments(self):
        """parse self.doc_opt and generates self.options"""
        for sec in self.cp.sections():
            for opt in self.cp.options(sec):
                cle = "%s_%s"%(sec,opt)
                cff = cfField(opt, self.cp.get(sec, opt))
                cff.doc = []
                cff.section = sec
                for l in self.doc_opt[opt]:
                    cff.subparse_comment(l)
                if Debug:
                    print cff
                    print
                self.options[cle] = cff

    def opt_templ(self, cff):
        "build the template for an option stored in cff"
        if cff.type_ != "%hidden%":
            doc = "<br/>\n".join(cff.doc)
            if cff.type_ in ("%text%", "%outfile%"):    # if string, add SIZE
                templ = '<div class="{2}"><p>{3}<br/><b>{{{{ form.{0}_{1}.label}}}}</b>: \
                    {{{{form.{0}_{1}(class="{2}",SIZE="60")}}}}</p></div>'.format( \
                                cff.section, cff.name, self.class_, doc)
            else:
                templ = '<div class="{2}"><p>{3}<br/><b>{{{{ form.{0}_{1}.label}}}}</b>: \
                    {{{{form.{0}_{1}(class="{2}")}}}}</p></div>'.format( \
                                cff.section, cff.name, self.class_, doc)
        else:   # special if hidden 
            templ = '{{{{form.{0}_{1}}}}}'.format(cff.section, cff.name)
        return templ

    def sect_templ(self, sec):
        "build the template for a section 'sec' "
        doc = "<br/>\n".join(self.doc_sec[sec])
        options_list = [self.opt_templ(self.options["%s_%s"%(sec,opt)])   for opt in self.cp.options(sec)] 
        options_templ = "  \n".join(options_list )
        sec_templ = """
<div class={0}.section>
<hr>
<h2>{1}</h2>
<b>{2}</b>
{3}
</div>
        """.format(self.class_, sec, doc, options_templ)
        return sec_templ

    def buildtemplate(self):
        "builds the jinja2 template string, from the config file"
        templ = [head]
        templ += ['<form method="{0}" action="{1}">'.format(self.method, self.callback)]
        for sec in self.cp.sections():
            templ.append( self.sect_templ(sec) )
        templ.append('<hr>\n    <input type="submit" name="submitform" value="Reload Values from file"/>')
        templ.append('    <input type="submit" name="submitform" value="Validate Values" />\n</form>')
        templ.append(foot)
        return "\n".join(templ)
    def buildforms(self):
        "build the wtform on the fly from the config file"
        dico = OrderedDict()
        values = {}
        for cle,content in self.options.items():        # construct the dict for dForm
            Field_type = cfFieldDef[content.type_][0]
            if content.type_ == "%select%":
                dico[cle] = Field_type(content.name, choices=zip(content.lopt,content.lopt), validators=content.validators)
            else:
                dico[cle] = Field_type(content.name, validators=content.validators)
            values[cle] = content.value
        df = dForm(dico)                    # create the form builder
        form = df(**values)                 # creat the form and load the values at once
        return form

    def render(self):
        'associate template and wtform to produce html'
        if self.form is None:                   # simple caching method
            self.form = self.buildforms()
        if self.template is None:               # simple caching method
            self.template = Template( self.buildtemplate() )
        html = self.template.render(form=self.form, title="ConfigForm", filename=self.cffilename)
        return html
    
    def produce(self):
        """produces the text of the current defined config file, as a list, one line per entry"""
        text = ["## File generated by ConfigForm v %s"%(__version__)]
        for sec in self.cp.sections():
            text.append('##########################################')
            text += ["# %s"%d for d in self.doc_sec[sec] ]
            text.append('[%s]'%sec)
            text.append('')
            for opt in self.cp.options(sec):
                cle = "%s_%s"%(sec,opt)
                text += ["# %s"%d for d in self.options[cle].doc ]
                if self.options[cle].meta is not None:
                    text.append('# %s'%(self.options[cle].meta) )
                text.append('%s = %s'%(opt,self.form[cle].data))
                text.append('')
        return text
    def writeback(self, filename):
        """writes back the config file with the current values"""
        with open(filename,'w') as F:
            F.write( "\n".join( self.produce() ) )
        
##########################
def BuildApp(cffile):
    app = Flask(__name__, static_url_path='/static')
    app.config['SECRET_KEY'] = 'kqjsdhf'
    cf = ConfigForm(cffile)
    cf.callback = '/callback'
    if Debug: print cf.doc_sec.keys()

    @app.route('/')
    def index():
        '''
        First page when launching the interface.
        '''
        return cf.render()
    @app.route('/callback',  methods = ['POST'])
    def callback():
        '''
        route called by the form - validate entries
        '''
        text = []
        if request.form["submitform"] == "Reload Values from file":
            cf.reload()
            return redirect(url_for('index'))
        else:   # Validate
            for cle,content in cf.options.items():
                type_ = cfFieldDef[content.type_][1]
                try:
                    cf.form[cle].data = type_((request.form[cle]))
                except:
                    cf.form[cle].data = request.form[cle]
                if Debug:
                    text.append("%s : %s<br/>"%(cle, request.form[cle]))
            if not cf.form.validate():  # error in validation
                text.append("<h1>Error in config file</h1><p>The following error are detected :</p>")
                for name,msges in cf.form.errors.items():
                    nmspl = name.split('_')
                    sec = nmspl[0]
                    opt = "_".join(nmspl[1:])
                    for msg in msges:
                        text.append('<li>in section <b>{}</b> entry <b>{}</b> : {}</li>\n'.format(sec,opt,msg))
                text.append("</ul>")
            else:
                text.append("<h1>Every thing is fine</h1>\n<p>All entries validated</p>")
                text.append('<p> <a href="{}">write file {}</a>'.format(url_for('write'), cf.cffilename) )
            text.append('<p> <a href="%s">back to form</a>'%(url_for('index'),) )
            return "\n".join(text)
    @app.route('/write')
    def write():
        tmpf = "_tempfile.cfg"
        while os.path.exists(tmpf):
            tmpf = "_tempfile_%d.cfg"%(int(1E6*random.random()))
        print tmpf
        cf.writeback(tmpf)      # write to tempfile
        os.rename(cf.cffilename, cf.cffilename+"~") # move away initial file
        os.rename(tmpf, cf.cffilename)
        return redirect(url_for('bye'))
    @app.route('/bye')
    def bye():
        'quitting'
        return "<H1>Bye !</H1>"
    return app

##########################
class ConfigFormTests(unittest.TestCase):
    "unittests"
    def setUp(self):
        self.verbose = 1    # verbose >0 switches on messages
    def announce(self):
        if self.verbose >0:
            print self.shortDescription()
            print "----------"
    def _test_dForm(self):
        """testing dForm"""
        self.announce()
        otest = OrderedDict()
        otest['urQRd'] = wtforms.SelectField('do urQRd processing', choices=[(True,'yes'),(False,'no')] )
        otest['urank'] = wtforms.IntegerField('choose urQRd rank',[validators.NumberRange(min=4, max=100)])
        otest['File'] = wtforms.FileField('File to process')
        df = dForm(otest)
        form2 = df(urank=123)
        temp = dTemplate(otest, "POST", "/param", "classX")
        html = temp.render(form=form2)
        print html
        self.assertTrue(html.startswith('<form method="POST" action="/param">') )
    def test_render(self):
        "test render"
        filename = "/Users/mad/NPKV2/pasteur_interface/QM/test.cfg"
        cf = ConfigForm(filename)
        print cf.doc_sec.keys()
        print cf.render()

##########################
def main(argv=None):
    "called at start-up"
    if argv is None:
        argv = sys.argv
    try:
        filename = argv[1]
    except:
        print >> sys.stderr, "wrong input"
        print >> sys.stderr, "Usage: python ConfigForm.py configfile.cfg"
        return 2
    print "Processing ",filename
    input_file = open(filename).readlines() # read the file - a bad quick hack for checking the file is ok.
    url = "http://{0}:{1}".format(HOST, PORT)
    app = BuildApp(filename)
    if StartWeb:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start() # open a page in the browser.
    try:
        app.run(port = PORT, debug = Debug)
    except socket.error:
        print >> sys.stderr, "wrong port number"
        print >> sys.stderr, "try using another port number (defined in the application header)"
        return 3
    return 0

if __name__ == '__main__':
#    unittest.main()
    sys.exit(main())
    