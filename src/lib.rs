/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use euclid::default::Size2D;

use fnv::FnvHashMap;

use std::hash::Hash;
use std::mem;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::RwLock;
use std::sync::RwLockReadGuard;
use std::sync::RwLockWriteGuard;

use surfman::Context;
use surfman::ContextID;
use surfman::Device;
use surfman::Error;
use surfman::Surface;

struct SwapChainData {
    size: Size2D<i32>,
    context_id: ContextID,
    unattached_front_buffer: Option<Surface>,
    pending_surface: Option<Surface>,
    presented_surfaces: Vec<Surface>,
}

impl SwapChainData {
    fn validate_context(&self, context: &mut Context) -> Result<(), Error> {
        if self.context_id == context.id() {
            Ok(())
        } else {
            Err(Error::IncompatibleContext)
        }
    }

    fn swap_buffers(&mut self, device: &mut Device, context: &mut Context) -> Result<(), Error> {
        self.validate_context(context)?;

        // Fetch a new back buffer, recycling presented buffers if possible.
        let new_back_buffer = self
            .presented_surfaces
            .iter()
            .position(|surface| surface.size() == self.size)
            .map(|index| Ok(self.presented_surfaces.swap_remove(index)))
            .unwrap_or_else(|| device.create_surface(context, &self.size))?;

        // Swap the buffers
        let new_front_buffer = match self.unattached_front_buffer.as_mut() {
            Some(surface) => mem::replace(surface, new_back_buffer),
            None => device.replace_context_surface(context, new_back_buffer)?,
        };

        // Updata the state
        self.pending_surface = Some(new_front_buffer);
        for surface in self.presented_surfaces.drain(..) {
            device.destroy_surface(context, surface)?;
        }

        Ok(())
    }

    fn take_surface(&mut self) -> Option<Surface> {
        self.pending_surface
            .take()
            .or_else(|| self.presented_surfaces.pop())
    }

    fn recycle_surface(&mut self, surface: Surface) {
        self.presented_surfaces.push(surface)
    }
}

#[derive(Clone)]
pub struct SwapChain(Arc<Mutex<SwapChainData>>);

impl SwapChain {
    fn lock(&self) -> MutexGuard<SwapChainData> {
        self.0.lock().unwrap_or_else(|err| err.into_inner())
    }

    pub fn swap_buffers(&self, device: &mut Device, context: &mut Context) -> Result<(), Error> {
        self.lock().swap_buffers(device, context)
    }

    pub fn take_surface(&self) -> Option<Surface> {
        self.lock().take_surface()
    }

    pub fn recycle_surface(&self, surface: Surface) {
        self.lock().recycle_surface(surface)
    }
}

#[derive(Clone, Default)]
pub struct SwapChains<SwapChainID: Eq + Hash> {
    table: Arc<RwLock<FnvHashMap<SwapChainID, SwapChain>>>,
}

impl<SwapChainID: Eq + Hash> SwapChains<SwapChainID> {
    fn table(&self) -> RwLockReadGuard<FnvHashMap<SwapChainID, SwapChain>> {
        self.table.read().unwrap_or_else(|err| err.into_inner())
    }

    fn table_mut(&self) -> RwLockWriteGuard<FnvHashMap<SwapChainID, SwapChain>> {
        self.table.write().unwrap_or_else(|err| err.into_inner())
    }

    pub fn get(&self, id: SwapChainID) -> Option<SwapChain> {
        self.table().get(&id).cloned()
    }

    pub fn get_with<F, T>(&self, id: SwapChainID, f: F) -> Option<T>
    where
        F: Fn(&SwapChain) -> T,
    {
        self.table().get(&id).map(f)
    }

    pub fn get_or_default(&self, id: SwapChainID, device: &Device, context: &mut Context) -> SwapChain {
        self.table_mut().entry(id).or_insert_with(move || {
	    let size = device.context_surface_size(context).unwrap();
            SwapChain(Arc::new(Mutex::new(SwapChainData {
                size,
                context_id: context.id(),
                unattached_front_buffer: None,
                pending_surface: None,
                presented_surfaces: Vec::new(),
            })))
        }).clone()
    }

    pub fn get_or_create(&self, id: SwapChainID, device: &mut Device, context: &mut Context, size: Size2D<i32>) -> SwapChain {
        self.table_mut().entry(id).or_insert_with(move || {
	    let surface = device.create_surface(context, &size).unwrap();
            SwapChain(Arc::new(Mutex::new(SwapChainData {
                size,
                context_id: context.id(),
                unattached_front_buffer: Some(surface),
                pending_surface: None,
                presented_surfaces: Vec::new(),
            })))
        }).clone()
    }
}
