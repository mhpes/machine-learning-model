import React, { useState, useEffect } from 'react';
import Card from '@material-ui/core/Card';
import CardContent from '@material-ui/core/CardContent';import './App.css';
import axios from 'axios';
import FileSaver from 'file-saver';
import { Typography, TextField, Grid, makeStyles, CardHeader, Button, Input } from '@material-ui/core';

const useStyles = makeStyles({
  root: {
      flexGrow:1,
      alignContent: "center",
      alignItems: "center",
      // alignContent: "center"
  },
  child: {
    width: "40%"
  }
});

const usePredictProduct = (product: String) => {
  const [data, setData] = useState();
  useEffect(() => {
    console.log(product)
    axios.get("http://localhost:5000/predictProduct/" + product).then(response => {
      setData(response.data.prediction)
    })
  }, [product])

  return data
}




const App = () => {
  const [product, setProduct] = useState('');
  const [file, setFile] = useState<File>();
  const [filename, setFilename] = useState('');

  const onSubmit = (event: any) => {
    event.preventDefault();
    const formData = new FormData();
    formData.append('data', file!);
    formData.append('filename', filename)
    
    axios.post('http://localhost:5000/predictCsv', formData, {
      headers:{
        'Content-Type': 'text/csv'
      }
    }).then(function(response) {
      console.log(response)
        setFile(response.data)
        setFilename(filename.substring(0, filename.length - 4) + '_response.csv')
        return new Blob([response.data], { type: "text/plain;charset=utf-8" });
      }).then(function(blob) {
        FileSaver.saveAs(blob, filename.substring(0, filename.length - 4) + '_response.csv');
      })
  }
  
  const onChange = (e:any) => {
    if(e.target.files[0] != null) {
      setFile(e.target.files[0]);
      setFilename(e.target.files[0].name);
    } else {
      setFile(undefined);
      setFilename('');
    }

  }

  const classes = useStyles();

  return (
      <div className="App">
        <header className="App-header">
        <Grid container direction="column" className={classes.root} spacing={10}>
          <Grid item xs={12} className={classes.child}>
            <Grid container justify="center" spacing={5}>
              <Card className="Main-Card">
                <CardHeader title="Product Suggester" />
                <CardContent>
                  <Grid container direction="column" justify="space-between" alignItems="flex-start" spacing={0}>
                  <TextField id="product" margin="normal" fullWidth label="Product" variant="outlined" onChange={e => setProduct(e.target.value)}/>
                  <Typography variant="h5">
                    Suggestion: {usePredictProduct(product)}
                  </Typography>
                  </Grid>
                </CardContent>
              </Card>
          </Grid>
          </Grid>
          <Grid item xs={12} className={classes.child}>
            <Grid container justify="center" spacing={5}>
              <Card className="Main-Card">
              <CardHeader title="Csv Suggester" />
                <CardContent>
                <form onSubmit={onSubmit}>
                  <Input type="file" onChange={onChange}/>
                  <Button variant="contained" type="submit" color="primary">
                  Submit
                </Button>
                </form>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Grid>
      </header>
    </div>  
  );
}

export default App;