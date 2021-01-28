# Python 3 server example
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket
import sys
from urllib import parse
login_html = """
<html>
    <head>
        <meta charset='utf-8'>
        <title>login</title>
        <style>
        </style>
    </head>
    <body>
        <fieldset style="width:250px;">
            <legend>login</legend>
                <form method='post'>
                    <p>pw : <input name='pw' type='password'> </p>
                    <input type='submit' value='login'>
                </form>
        </fieldset>
    </body>
</html>
"""

graph_top_html = """
<!DOCTYPE HTML>
<html>
<head>
<style>html, body { width: 100%; height: 100%; margin: 0;}.container { position: relative; width: 100%; height: 100%; }</style>
<script type="text/javascript">
window.onload = function () {
var chart = new CanvasJS.Chart("chartContainer", {
    theme: "light2",
    title: { text: '"""

graph_mid_html1 = """' },
    axisX: { viewportMinimum: 0.99, gridThickness: 1 },
    data: [{ type: "line", indexLabelFontSize: 16, dataPoints: [
"""

graph_mid_html2 = """
]}]});
chart.render();}
</script>
</head>

<body>
<div class="container" id="chartContainer" style="height: 30em; width: 100%;"></div>
<script src="https://canvasjs.com/assets/script/canvasjs.min.js"></script>
<div class="container" style="height: 30em; width: 100%;">
    <input type="button" id="auto_refresh" value="Auto Refresh" style="font-size:2em; float:right" onclick="auto_refresh();">
    <input type="button" id="refresh" value="Refresh" style="font-size:2em; float:right;" onclick="refresh();">
    <span style = "font-size:2em;">Refresh Time (sec)</span> <input type="text" id="refresh_time" value="" style="font-size:2em;" onchange="refresh_time();">
</div>

<script type="text/javascript">
"""

graph_bot_html = """
document.getElementById('refresh_time').value = refresh_time_int;
if(auto_refresh_bool){
    document.getElementById('auto_refresh').style.backgroundColor='rgb(150,255,150)';
    loop=setTimeout(send_post, refresh_time_int*1000, "refresh", auto_refresh_bool, "refresh_time", refresh_time_int);
}
else{
    document.getElementById('auto_refresh').style.backgroundColor='rgb(255,150,150)';
}

function send_post(n, v){
    str = '<form id="smb_form" method="post">';
    for(i=0; i<arguments.length/2; i++){
        str += '<input type="hidden" name="' + arguments[i*2] + '" value="' + arguments[i*2+1]+ '">';
    }
    str += '</form>';
    document.write(str);
    document.getElementById("smb_form").submit()
}
function refresh() {
    send_post("refresh", auto_refresh_bool, "refresh_time", refresh_time_int);
}
function auto_refresh() {
    if(auto_refresh_bool){
        auto_refresh_bool=false;
        clearTimeout(loop);
        document.getElementById('auto_refresh').style.backgroundColor='rgb(255,150,150)';
    }
    else{
        auto_refresh_bool=true;
        document.getElementById('auto_refresh').style.backgroundColor='rgb(150,255,150)';
        send_post("refresh", auto_refresh_bool, "refresh_time", refresh_time_int)
    }
}
function refresh_time(){
    refresh_time_int = Number(document.getElementById('refresh_time').value);
    if (refresh_time_int<1){
        alert("Refresh time must larger than 1 second");
        return;
    }
    clearTimeout(loop);
    loop=setTimeout(send_post, refresh_time_int*1000, "refresh", auto_refresh_bool, "refresh_time", refresh_time_int);
}
</script>

</body>
</html>
"""


class MyServer(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        self._set_response()
        self.path = parse.unquote(self.path)
        if self.path == "/":
            self.wfile.write(bytes(
                '<strong>Insert {}:{}/<*.plot file path> to address bar</strong><br><br>'.format(hostName, serverPort), 'utf8'))
            self.wfile.write(bytes('Ex) {}:{}/home/graph.plot<br><br><br>'.format(hostName, serverPort), 'utf8'))
            self.wfile.write(bytes('<strong>(*.plot) file example</strong><br><br>', 'utf8'))
            self.wfile.write(bytes('Ex) {x:1, y:0.01},{x:2, y:0.02},{x:3, y:0.03}, ... ,{x:index, y:value},', 'utf8'))
        else:
            if self.check_file(self.path[1:], "plot"):
                self.wfile.write(bytes(login_html, 'utf8'))
            else:
                self.wfile.write(bytes('<strong>Insert {}:{}/<*.plot file path> to address bar</strong><br><br>'.format(hostName, serverPort), 'utf8'))
                self.wfile.write(bytes('Ex) {}:{}/home/graph.plot<br><br><br>'.format(hostName, serverPort), 'utf8'))
                self.wfile.write(bytes('<strong>(*.plot) file example</strong><br><br>', 'utf8'))
                self.wfile.write(bytes('Ex) {x:1, y:0.01},{x:2, y:0.02},{x:3, y:0.03}, ... ,{x:index, y:value},', 'utf8'))

    def do_POST(self):
        self._set_response()
        self.path = parse.unquote(self.path)
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf8').split('&')
        post_data = [s.split('=') for s in post_data]
        if post_data[0][0] == 'pw':
            if post_data[0][1] != password:
                self.message_window("login failed")
                self.wfile.write(bytes(login_html, 'utf8'))
                return
            self.draw_graph(self.path[1:])
        if post_data[0][0] == 'refresh':
            self.draw_graph(self.path[1:], post_data[0][1], int(post_data[1][1]))

    def send_file(self, filename):
        try:
            with open(filename) as f:
                self.wfile.write(bytes(f.read(), 'utf8'))
        except FileNotFoundError:
            self.message_window("Couldn't open file")

    def check_file(self, filename, extension=""):
        if not '.' in filename:
            self.message_window("File path is incorrect")
            return False
        if extension != '' and filename.split('.')[1] != extension:
            self.message_window("Extension is incorrect. It must be (*.{})".format(extension))
            return False
        try:
            with open(filename) as f:
                pass
        except FileNotFoundError:
            self.message_window("File path is incorrect")
            return False
        return True

    def message_window(self, message):
        self.wfile.write(bytes('<script type = "text/javascript">', 'utf8'))
        self.wfile.write(bytes('alert("{}")'.format(message), 'utf8'))
        self.wfile.write(bytes('</script>', 'utf8'))

    def draw_graph(self, filename, auto_refresh="true", refresh_time_int=10):
        self.wfile.write(bytes(graph_top_html, 'utf8'))
        graph_name = filename.split('/')
        graph_name = graph_name[-1].split('.')[0]
        self.wfile.write(bytes(graph_name, 'utf8'))
        self.wfile.write(bytes(graph_mid_html1, 'utf8'))
        self.send_file(filename)
        self.wfile.write(bytes(graph_mid_html2, 'utf8'))
        self.wfile.write(bytes('var auto_refresh_bool = {};'.format(auto_refresh), 'utf8'))
        self.wfile.write(bytes('var refresh_time_int = {};'.format(refresh_time_int), 'utf8'))
        self.wfile.write(bytes(graph_bot_html, 'utf8'))

if __name__ == "__main__":
    serverPort = 8080
    password = "1234"
    hostName = "0.0.0.0"

    print("Web graph plot Server")
    print()
    print("< Program options >")
    print("-P : change passward (default:{})".format(password))
    print("-p : change port number (default:{})".format(serverPort))
    print("-I : change IP address (default:auto)")
    print("If you want to change parameter, re run program with <$sudo python WGBserver.py -P (passward) -p (portnumber)> -I (ip-address)")
    print()

    custom_IP_flag = False
    if(len(sys.argv)!=1):
        for i in range(1, len(sys.argv), 2):
            try:
                if sys.argv[i] == '-P':
                    password = sys.argv[i+1]
                elif sys.argv[i] == '-p':
                    serverPort = int(sys.argv[i+1])
                elif sys.argv[i] == '-I':
                    hostName = sys.argv[i+1]
                    custom_IP_flag = True
                else:
                    print("There is no command <{}>".format(sys.argv[i]))
                    sys.exit(1)
            except IndexError:
                print("Need more arguments in command <{}>".format(sys.argv[i]))
                sys.exit(2)
            except ValueError:
                print("Port number must be a number <{}>".format(sys.argv[i+1]))
                sys.exit(3)

    if not custom_IP_flag:
        ip_list = socket.gethostbyname_ex(socket.gethostname())[2]
        if len(ip_list) != 1:
            print("Select your IP")
            for i in range(len(ip_list)):
                print("{}. {}".format(i, ip_list[i]))

            select_ip = input('>>')

            while not select_ip.isnumeric() or not (0 <= int(select_ip) < len(ip_list)):
                print("Incorrect number, Try again")
                select_ip = input('>>')
            ip_list = ip_list[int(select_ip)]
            print()
        else:
            ip_list = ip_list[0]
        hostName = ip_list

    try:
        webServer = ThreadingHTTPServer((hostName, serverPort), MyServer)
    except:
        print("Couldn't open server. Check IP-address({}) and port number({})".format(hostName, serverPort))
        sys.exit(4)

    print("Server started http://{}:{} <passward:{}>, use <Ctrl-C> to stop".format(hostName, serverPort, password))
    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
