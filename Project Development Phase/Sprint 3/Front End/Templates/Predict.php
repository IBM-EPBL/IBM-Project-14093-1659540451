
<!doctype html>
<html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.4.1/dist/css/bootstrap.min.css">
        <title>
            Page of Detection
        </title>
        <link rel="stylesheet" href="./front.css">
        <script>
            function clearFunc()
            {
            document.getElementById("name").value="";
            document.getElementById("dob").value="";
            document.getElementById("gender").value="";
            document.getElementById("address").value="";
            document.getElementById("spiral").value="";
            document.getElementById("wave").value="";
        }
        </script>
    </head>
    
    <body id="bg">  
        <div class="container">
            <form method="post">
                <div class="form-group col-md-5">
                    <h5> PATIENTS FORM </h5>
                    <label>Enter Your Name</label>
                    <input type="text" class="form-control" id="Name" name="name" placeholder="Enter Your Name" >
                </div>

                <div class="form-group col-md-5"> 
                    <label>Enter Your Date of Birth</label>
                    <input type="date" class="form-control" id="dob" name="dob" placeholder="Enter Your Date of Birth" >
                </div>

                <div class="form-group col-md-5">
                    <label for="gender">Gender</label>
                    <select class="form-control" id="gender" name="gender">
                    <option value="">-- Choose Your Gender --</option>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                    <option value="Transgender">Transgender</option>
                    </select>
                </div>

                <div class="form-group col-md-5"> 
                    <label>Enter Your Phone Number</label>
                    <input type="varchar" class="form-control" id="blood" name="phone" placeholder="+91 9361X XXXXX" >
                </div>

                <div class="form-row">
                &nbsp;&nbsp;&nbsp;&nbsp;<div class="form-group col-md-2"> 
                    <label>Enter Your Blood Group</label>
                    <input type="varchar" class="form-control" id="blood" name="blood" placeholder="Ex. B+ve" >
                </div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

                <div class="form-group col-md-1.5"> 
                    <label>Age</label>
                    <input type="varchar" class="form-control" id="age" name="age" placeholder="Enter Your Age" >
                </div></div>
                
                <div class="form-group col-md-5"> 
                    <label>Enter Your Address</label>
                    <input type="varchar" class="form-control" id="address" name="address" placeholder="Enter Your Address" >
                </div>

                <div class="form-row">&nbsp;&nbsp;&nbsp;&nbsp;
                <div class="form-group col-md-3"> 
                    <label>Upload The Spiral Image</label>
                    <input type="file" class="form-control-file" id="spiral" name="spiral">
                </div>
        
                <div class="form-group col-md-3"> 
                    <label>Upload The Wave Image</label>
                    <input type="file" class="form-control-file" id="wave" name="wave">
                </div>
                </div>

                <div class="form-row">
                &nbsp;&nbsp;&nbsp;&nbsp;<div><button type="submit" class="btn btn-primary" name="submit">Submit</button></div>&nbsp;&nbsp;&nbsp;&nbsp;
                    <div><button class="btn btn-primary" type="reset" onclick="clearFunc()">Reset</button></div>
                </div>
            </form>
        </div>
    </body>
</html>